#!/usr/bin/env python3
"""Simple web UI for the CAD cutter pipeline."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import logging
import math
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np
import trimesh
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform, BRepBuilderAPI_Transform
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.gp import gp_Ax1, gp_Dir, gp_GTrsf, gp_Pnt, gp_Trsf

import assemble
from assemble import progress

SUPPORTED_EXT = assemble.ALL_EXTENSIONS
GRADIENT_EXT = {".wrl", ".vrml", ".stl", ".3mf"}
SUPPORTED_WORKFLOWS = {
    "cylinder": "Cylinder",
    "generic": "Generic Stack",
}


def _safe_resolve(user_path: str) -> Path:
    """Resolve a user-supplied path and ensure it stays within cwd.

    Raises ValueError if the resolved path escapes the working directory.
    """
    cwd = Path.cwd().resolve()
    resolved = (cwd / user_path).resolve()
    if not str(resolved).startswith(str(cwd)):
        raise ValueError(f"Path escapes working directory: {user_path}")
    return resolved


def _relative_to_cwd(path: Path) -> str:
    """Return a cwd-relative path string, keeping the root as '.'."""
    cwd = Path.cwd().resolve()
    rel = path.resolve().relative_to(cwd)
    rel_str = rel.as_posix()
    return rel_str or "."


def _mesh_file_url(path: str | None) -> str | None:
    if not path:
        return None
    try:
        rel = _relative_to_cwd(Path(path))
    except Exception:
        return None
    return f"/api/mesh-file?path={rel}"


def _preview_source_path_for_part(part: "PartState") -> str | None:
    return part.preview_path or part.mesh_source_path or part.file_path


@dataclass
class PartState:
    file_path: str
    name: str
    source_ext: str
    shape: Any
    mesh_source_path: str | None = None
    preview_path: str | None = None
    preview_ext: str | None = None
    preview_only: bool = False
    material: str | None = None
    auto_oriented: bool = False
    orientation_steps: list[tuple[tuple[float, float, float], float]] = field(default_factory=list)
    settle_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    manual_scale: float = 1.0


@dataclass
class SessionState:
    parts: list[PartState] = field(default_factory=list)
    parts_dir: Path = field(default_factory=lambda: Path.cwd().resolve())
    gap: float = 0.0
    axis: str = "z"
    workflow: str = "cylinder"
    fine_orient: bool = False
    cut_angle: float = 90.0
    section_number: int | None = None


state = SessionState()
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

_SHAPE_CACHE: dict[tuple[str, float, bool], tuple[Any, str]] = {}
_MESH_PAYLOAD_CACHE: dict[tuple[str, float, int, float], dict[str, Any]] = {}
_DECIMATED_PAYLOAD_CACHE: dict[tuple[str, float, int, float, int], dict[str, Any]] = {}
_PREVIEW_SURROGATE_MIN_BYTES = 25 * 1024 * 1024
_WEBUI_CACHE_DIR = Path(".webui_cache")
_DEBUG_LOG: collections.deque[dict[str, Any]] = collections.deque(maxlen=400)
_DEBUG_LOG_LOCK = threading.Lock()
_DEBUG_LOG_PATH = _WEBUI_CACHE_DIR / "debug_log.jsonl"
_THUMB_TRIANGLE_TARGET = 2500
_ORIENT_TRIANGLE_TARGET = 4000
_ORIENT_VERTEX_LIMIT = 12000


def _compute_webui_version() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "--", "web_ui.py", "static/app.js", "static/style.css", "templates/index.html"],
            cwd=Path(__file__).resolve().parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode != 0
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return f"mtime-{int(Path(__file__).stat().st_mtime)}"


_WEBUI_VERSION = _compute_webui_version()


def _record_debug_log(source: str, kind: str, message: str, meta: dict[str, Any] | None = None) -> None:
    entry = {
        "ts": time.time(),
        "source": source,
        "kind": kind,
        "message": message,
        "meta": meta or {},
    }
    with _DEBUG_LOG_LOCK:
        _DEBUG_LOG.append(entry)
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def _debug_log_snapshot() -> list[dict[str, Any]]:
    with _DEBUG_LOG_LOCK:
        return list(_DEBUG_LOG)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _mesh_payload(shape, tolerance: float = 0.5) -> dict[str, Any]:
    """Tessellate a shape and return a dict of flat vertex/index lists."""
    verts, faces = assemble.tessellate_shape(shape, tolerance=tolerance)
    if len(verts) == 0:
        return {"vertices": [], "indices": []}
    indices = faces[:, 1:].reshape(-1).tolist() if len(faces) else []
    return {"vertices": verts.reshape(-1).tolist(), "indices": indices}


def _payload_cache_path(path: Path, tolerance: float = 0.5) -> Path:
    stat = path.stat()
    key = "|".join((
        "mesh-payload-v1",
        str(path.resolve()),
        str(stat.st_mtime_ns),
        str(stat.st_size),
        f"{tolerance:.6f}",
    ))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return _WEBUI_CACHE_DIR / "mesh_payloads" / f"{digest}.json.gz"


def _decimated_payload_cache_path(path: Path, tolerance: float, target_triangles: int) -> Path:
    stat = path.stat()
    key = "|".join((
        "mesh-decimated-v1",
        str(path.resolve()),
        str(stat.st_mtime_ns),
        str(stat.st_size),
        f"{tolerance:.6f}",
        str(target_triangles),
    ))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return _WEBUI_CACHE_DIR / "mesh_payloads" / f"{digest}.json.gz"


def _payload_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(payload.get("vertices", []), dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(payload.get("indices", []), dtype=np.int64).reshape(-1, 3)
    return vertices, indices


def _payload_from_arrays(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    return {
        "vertices": np.asarray(vertices, dtype=np.float32).reshape(-1).tolist(),
        "indices": np.asarray(faces, dtype=np.int64).reshape(-1).tolist(),
    }


def _decimate_payload(payload: dict[str, Any], target_triangles: int) -> dict[str, Any]:
    vertices, faces = _payload_arrays(payload)
    if len(faces) <= target_triangles or len(vertices) == 0:
        return payload

    mesh = None
    try:
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64), process=False)
        simplify = getattr(mesh, "simplify_quadric_decimation", None) or getattr(mesh, "simplify_quadratic_decimation", None)
        if callable(simplify):
            try:
                mesh = simplify(target_triangles)
            except TypeError:
                mesh = simplify(face_count=target_triangles)
    except Exception:
        mesh = None

    if mesh is None or len(getattr(mesh, "faces", [])) == 0 or len(mesh.faces) >= len(faces):
        step = max(1, math.ceil(len(faces) / target_triangles))
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces[::step], dtype=np.int64), process=False)

    mesh.remove_unreferenced_vertices()
    return _payload_from_arrays(np.asarray(mesh.vertices), np.asarray(mesh.faces))


def _load_decimated_mesh_payload(
    source_path: str | None,
    shape,
    *,
    tolerance: float = 0.5,
    target_triangles: int = _THUMB_TRIANGLE_TARGET,
) -> dict[str, Any]:
    if not source_path:
        return _decimate_payload(_mesh_payload(shape, tolerance=tolerance), target_triangles)

    path = Path(source_path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, tolerance, target_triangles)
    cached = _DECIMATED_PAYLOAD_CACHE.get(key)
    if cached is not None:
        return cached

    cache_path = _decimated_payload_cache_path(path, tolerance, target_triangles)
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        _DECIMATED_PAYLOAD_CACHE[key] = payload
        return payload

    full_payload = _load_mesh_payload_from_cache_file(source_path, shape, tolerance=tolerance)
    payload = _decimate_payload(full_payload, target_triangles)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    _DECIMATED_PAYLOAD_CACHE[key] = payload
    return payload


def _load_mesh_payload_from_cache_file(
    source_path: str | None,
    shape,
    tolerance: float = 0.5,
) -> dict[str, Any]:
    if not source_path:
        return _mesh_payload(shape, tolerance=tolerance)

    path = Path(source_path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, tolerance)
    cached = _MESH_PAYLOAD_CACHE.get(key)
    if cached is not None:
        return cached

    cache_path = _payload_cache_path(path, tolerance=tolerance)
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        _MESH_PAYLOAD_CACHE[key] = payload
        return payload

    if path.suffix.lower() in assemble.MESH_EXTENSIONS:
        mesh = trimesh.load(str(path), force="mesh")
        payload = {
            "vertices": mesh.vertices.reshape(-1).tolist(),
            "indices": mesh.faces.reshape(-1).tolist(),
        }
    else:
        payload = _mesh_payload(shape, tolerance=tolerance)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    _MESH_PAYLOAD_CACHE[key] = payload
    return payload


def _count_faces(shape) -> int:
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


class _SlowProgressDetail:
    """Emit increasingly detailed status updates while a step is slow."""

    def __init__(self, initial_message: str, detail_factory):
        self.initial_message = initial_message
        self.detail_factory = detail_factory
        self._stop = threading.Event()
        self._thread = None
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        progress.note(self.initial_message)

        def worker():
            if self._stop.wait(1.0):
                return
            while not self._stop.is_set():
                elapsed = time.time() - self._start
                progress.note(self.detail_factory(elapsed))
                if self._stop.wait(1.0):
                    return

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)


def _rotate_shape(shape, rx: float, ry: float, rz: float):
    wp = cq.Workplane("XY").newObject([cq.Shape(shape)])
    if rx:
        wp = wp.rotate((0, 0, 0), (1, 0, 0), rx)
    if ry:
        wp = wp.rotate((0, 0, 0), (0, 1, 0), ry)
    if rz:
        wp = wp.rotate((0, 0, 0), (0, 0, 1), rz)
    return wp.val().wrapped


def _axis_angle_transform(shape, axis: tuple[float, float, float], angle_rad: float):
    if abs(angle_rad) < 1e-9:
        return shape
    ax = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(ax))
    if norm < 1e-9:
        return shape
    ax = ax / norm
    trsf = gp_Trsf()
    trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(*ax.tolist())), angle_rad)
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _scale_shape(shape, factor: float):
    g = gp_GTrsf()
    g.SetValue(1, 1, factor)
    g.SetValue(2, 2, factor)
    g.SetValue(3, 3, factor)
    return BRepBuilderAPI_GTransform(shape, g, True).Shape()


def _apply_manual_transforms(part: PartState):
    shape = part.shape
    for axis, angle_rad in part.orientation_steps:
        shape = _axis_angle_transform(shape, axis, angle_rad)
    rx, ry, rz = part.rot_xyz
    shape = _rotate_shape(shape, rx, ry, rz)
    if abs(part.manual_scale - 1.0) > 1e-6:
        shape = _scale_shape(shape, part.manual_scale)
    return shape


def _part_has_runtime_transform(part: PartState) -> bool:
    return (
        bool(part.orientation_steps)
        or part.rot_xyz != (0.0, 0.0, 0.0)
        or abs(part.manual_scale - 1.0) > 1e-6
    )


def _serialize_orientation_steps(steps: list[tuple[tuple[float, float, float], float]]) -> list[dict[str, Any]]:
    return [
        {"axis": [float(axis[0]), float(axis[1]), float(axis[2])], "angle": float(angle_rad)}
        for axis, angle_rad in steps
    ]


def _part_display_name(part: PartState) -> str:
    return part.name if not part.material else f"{part.name}_{part.material}"


def _location_to_offset(loc) -> tuple[float, float, float]:
    coords = loc.toTuple()[0]
    return (float(coords[0]), float(coords[1]), float(coords[2]))


def _apply_settle_offsets(part_info):
    offsets_by_name: dict[str, list[tuple[float, float, float]]] = {}
    for part in state.parts:
        offsets_by_name.setdefault(_part_display_name(part), []).append(part.settle_offset)

    adjusted = []
    for entry in part_info:
        name, shape, loc, rgb = entry[:4]
        rest = entry[4:]
        offsets = offsets_by_name.get(name)
        settle = offsets.pop(0) if offsets else (0.0, 0.0, 0.0)
        if any(abs(v) > 1e-9 for v in settle):
            loc = assemble.Location(assemble.Vector(*settle)) * loc
        adjusted.append((name, shape, loc, rgb, *rest))
    return adjusted


def _store_settle_offsets(base_part_info, updated_part_info) -> None:
    offsets_by_name: dict[str, list[tuple[float, float, float]]] = {}
    for base, updated in zip(base_part_info, updated_part_info):
        base_name, _shape, base_loc = base[:3]
        updated_loc = updated[2]
        base_offset = np.asarray(_location_to_offset(base_loc), dtype=float)
        updated_offset = np.asarray(_location_to_offset(updated_loc), dtype=float)
        delta = tuple((updated_offset - base_offset).tolist())
        offsets_by_name.setdefault(base_name, []).append(delta)

    for part in state.parts:
        key = _part_display_name(part)
        queue = offsets_by_name.get(key)
        part.settle_offset = queue.pop(0) if queue else (0.0, 0.0, 0.0)


def _parts_for_stack() -> list[tuple[cq.Workplane, str, str, bool]]:
    entries = []
    for idx, p in enumerate(state.parts):
        transformed = _apply_manual_transforms(p)
        wp = cq.Workplane("XY").newObject([cq.Shape(transformed)])
        name = p.name if not p.material else f"{p.name}_{p.material}"
        entries.append((wp, name, p.source_ext, assemble.is_mesh_file(p.file_path), idx))
    return entries


def _parts_for_assembly() -> list[tuple[cq.Workplane, str]]:
    """Return the subset of part data expected by assemble.stack_parts()."""
    return [(wp, name) for wp, name, *_ in _parts_for_stack()]


def _parts_for_export() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in state.parts:
        out.append({
            "name": _part_display_name(part),
            "shape": _apply_manual_transforms(part),
            "source_ext": part.source_ext,
            "is_mesh": assemble.is_mesh_file(part.file_path),
        })
    return out


def _export_parts_fast(output_dir: str) -> None:
    import os
    import trimesh

    os.makedirs(output_dir, exist_ok=True)
    used_names: dict[str, int] = {}
    for part in state.parts:
        name = _part_display_name(part)
        suffix = used_names.get(name, 0)
        used_names[name] = suffix + 1
        out_name = f"{name}_{suffix}{part.source_ext}" if suffix else f"{name}{part.source_ext}"
        out_path = Path(output_dir) / out_name

        if assemble.is_mesh_file(part.file_path):
            payload = _load_mesh_payload_from_cache_file(part.file_path, part.shape)
            vertices, faces = _payload_arrays(payload)
            mesh = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=float),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            mesh.apply_transform(_part_transform_matrix_4x4(part))
            mesh.export(out_path)
            continue

        _ensure_real_geometry(part)
        if out_path.suffix.lower() not in {".step", ".stp", ".iges", ".igs", ".brep"}:
            out_path = out_path.with_suffix(".step")
        assemble.export_part_native(_apply_manual_transforms(part), str(out_path), out_path.suffix, False)


def _mesh_bbox(payload: dict[str, Any]) -> tuple[list[float], list[float]]:
    verts = payload.get("vertices", [])
    if not verts:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    xs = verts[0::3]
    ys = verts[1::3]
    zs = verts[2::3]
    mins = [min(xs), min(ys), min(zs)]
    maxs = [max(xs), max(ys), max(zs)]
    return mins, maxs


def _axis_angle_matrix(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(ax))
    if norm < 1e-9 or abs(angle_rad) < 1e-9:
        return np.eye(3, dtype=float)
    ax = ax / norm
    x, y, z = ax.tolist()
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    return np.array([
        [t*x*x + c,     t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z,   t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y,   t*y*z + s*x, t*z*z + c],
    ], dtype=float)


def _euler_rotation_matrix_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    rx_rad = math.radians(rx)
    ry_rad = math.radians(ry)
    rz_rad = math.radians(rz)
    cx, sx = math.cos(rx_rad), math.sin(rx_rad)
    cy, sy = math.cos(ry_rad), math.sin(ry_rad)
    cz, sz = math.cos(rz_rad), math.sin(rz_rad)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return mz @ my @ mx


def _part_linear_transform_matrix(part: PartState) -> np.ndarray:
    matrix = np.eye(3, dtype=float)
    for axis, angle_rad in part.orientation_steps:
        matrix = _axis_angle_matrix(axis, angle_rad) @ matrix
    matrix = _euler_rotation_matrix_deg(*part.rot_xyz) @ matrix
    matrix = matrix * float(part.manual_scale)
    return matrix


def _transformed_bbox(mins: list[float], maxs: list[float], part: PartState) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([
        [x, y, z]
        for x in (mins[0], maxs[0])
        for y in (mins[1], maxs[1])
        for z in (mins[2], maxs[2])
    ], dtype=float)
    transformed = corners @ _part_linear_transform_matrix(part).T
    return transformed.min(axis=0), transformed.max(axis=0)


def _part_transform_matrix_4x4(part: PartState) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _part_linear_transform_matrix(part)
    return matrix


def _preview_payload_for_part(part: PartState) -> dict[str, Any]:
    source_path = _preview_source_path_for_part(part)
    return _load_mesh_payload_from_cache_file(source_path, part.shape)


def _preview_thumb_payload_for_part(part: PartState, target_triangles: int = _THUMB_TRIANGLE_TARGET) -> dict[str, Any]:
    source_path = _preview_source_path_for_part(part)
    return _load_decimated_mesh_payload(
        source_path,
        part.shape,
        tolerance=0.8,
        target_triangles=target_triangles,
    )


def _preview_bounds_for_part(part: PartState) -> tuple[np.ndarray, np.ndarray]:
    payload = _preview_payload_for_part(part)
    mins, maxs = _mesh_bbox(payload)
    return _transformed_bbox(mins, maxs, part)


def _axis_vector(axis_name: str) -> np.ndarray:
    axis = assemble.AXIS_MAP.get(axis_name, assemble.AXIS_MAP["z"])
    if hasattr(axis, "toTuple"):
        axis = axis.toTuple()
    elif hasattr(axis, "x") and hasattr(axis, "y") and hasattr(axis, "z"):
        axis = (axis.x, axis.y, axis.z)
    return np.asarray(axis, dtype=float)


def _perpendicular_axis(target: np.ndarray) -> np.ndarray:
    fallback = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(target, fallback))) > 0.9:
        fallback = np.array([0.0, 1.0, 0.0], dtype=float)
    perp = np.cross(target, fallback)
    norm = float(np.linalg.norm(perp))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return perp / norm


def _rotation_steps_to_axis(principal: np.ndarray, target: np.ndarray, conic_flip: bool) -> list[tuple[tuple[float, float, float], float]]:
    principal = principal / np.linalg.norm(principal)
    target = target / np.linalg.norm(target)
    dot = float(np.clip(np.dot(principal, target), -1.0, 1.0))
    steps: list[tuple[tuple[float, float, float], float]] = []

    if dot < 1.0 - 1e-6:
        if dot <= -1.0 + 1e-6:
            rot_axis = _perpendicular_axis(target)
        else:
            rot_axis = np.cross(principal, target)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
        steps.append((tuple(rot_axis.tolist()), math.acos(dot)))

    if conic_flip:
        steps.append((tuple(_perpendicular_axis(target).tolist()), math.pi))

    return steps


def _apply_steps_to_vertices(vertices: np.ndarray, steps: list[tuple[tuple[float, float, float], float]]) -> np.ndarray:
    if len(vertices) == 0 or not steps:
        return vertices
    out = np.asarray(vertices, dtype=float)
    for axis, angle in steps:
        ax = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(ax))
        if norm < 1e-9 or abs(angle) < 1e-9:
            continue
        ax = ax / norm
        x, y, z = ax.tolist()
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        rot = np.array([
            [t*x*x + c,     t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z,   t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y,   t*y*z + s*x, t*z*z + c],
        ], dtype=float)
        out = out @ rot.T
    return out


def _sample_vertices(payload: dict[str, Any], limit: int = _ORIENT_VERTEX_LIMIT) -> np.ndarray:
    vertices, _faces = _payload_arrays(payload)
    if len(vertices) <= limit:
        return vertices
    step = max(1, math.ceil(len(vertices) / limit))
    return vertices[::step]


def _orientation_payload_for_part(part: PartState, target_triangles: int = _ORIENT_TRIANGLE_TARGET) -> dict[str, Any]:
    source_path = _preview_source_path_for_part(part)
    if not _part_has_runtime_transform(part) and source_path:
        return _load_decimated_mesh_payload(
            source_path,
            part.shape,
            tolerance=0.8,
            target_triangles=target_triangles,
        )
    effective_shape = _apply_manual_transforms(part)
    return _decimate_payload(_mesh_payload(effective_shape, tolerance=0.8), target_triangles)


def _autodrop_mesh_payloads(part_info) -> list[dict[str, Any] | None]:
    payloads: list[dict[str, Any] | None] = []
    part_lookup: dict[str, list[PartState]] = {}
    for part in state.parts:
        part_lookup.setdefault(_part_display_name(part), []).append(part)

    for entry in part_info:
        name, shape, _loc = entry[:3]
        queue = part_lookup.get(name)
        part = queue.pop(0) if queue else None
        source_path = _preview_source_path_for_part(part) if part is not None else None
        if part is None or not source_path:
            payloads.append(None)
            continue
        try:
            payloads.append(_load_mesh_payload_from_cache_file(source_path, shape))
        except Exception:
            payloads.append(None)
    return payloads


def _can_use_fast_mesh_scene(part_states: list[PartState]) -> bool:
    return bool(part_states) and all(_preview_source_path_for_part(part) for part in part_states)


def _preview_stack_offsets(records: list[dict[str, Any]]) -> list[list[float]]:
    axis_index = {"x": 0, "y": 1, "z": 2}.get(state.axis, 2)
    center_axes = [idx for idx in range(3) if idx != axis_index]
    offsets: list[list[float]] = [[0.0, 0.0, 0.0] for _ in records]
    auto_level = 1
    classified = []
    levels_map: dict[int, dict[str, list[dict[str, Any]]]] = {}
    spanning: list[dict[str, Any]] = []

    for record in records:
        tier, levels, segment = assemble.parse_part_name(record["part"].name)
        if tier is None or not levels:
            tier = "outer"
            levels = [auto_level]
            segment = None
            auto_level += 1
        entry = {"tier": tier, "levels": levels, "segment": segment, "record": record}
        classified.append(entry)
        if len(levels) == 1:
            levels_map.setdefault(levels[0], {}).setdefault(tier, []).append(entry)
        else:
            spanning.append(entry)
            for level in levels:
                levels_map.setdefault(level, {})

    cursor = 0.0
    level_z_base: dict[int, float] = {}
    level_z_top: dict[int, float] = {}
    for level in sorted(levels_map):
        group = levels_map[level]
        level_z_base[level] = cursor
        if not group:
            level_z_top[level] = cursor
            continue

        ref_entry = None
        for tier_name in ("outer", "mid", "inner"):
            entries = group.get(tier_name)
            if entries:
                ref_entry = entries[0]
                break
        if ref_entry is None:
            level_z_top[level] = cursor
            continue

        ref_record = ref_entry["record"]
        ref_height = float(ref_record["maxs"][axis_index] - ref_record["mins"][axis_index])

        for tier_name in ("outer", "mid", "inner"):
            for entry in group.get(tier_name, []):
                record = entry["record"]
                mins = record["mins"]
                maxs = record["maxs"]
                offset = [0.0, 0.0, 0.0]
                for center_axis in center_axes:
                    offset[center_axis] = -((mins[center_axis] + maxs[center_axis]) / 2.0)
                offset[axis_index] = cursor - mins[axis_index]
                offsets[record["index"]] = offset

        level_z_top[level] = cursor + ref_height
        cursor += ref_height + state.gap

    for entry in spanning:
        levels = entry["levels"]
        first_level = min(levels)
        if first_level not in level_z_base:
            continue
        record = entry["record"]
        mins = record["mins"]
        maxs = record["maxs"]
        offset = [0.0, 0.0, 0.0]
        for center_axis in center_axes:
            offset[center_axis] = -((mins[center_axis] + maxs[center_axis]) / 2.0)
        offset[axis_index] = level_z_base[first_level] - mins[axis_index]
        offsets[record["index"]] = offset

    return offsets


def _preview_records() -> list[dict[str, Any]]:
    return [_preview_record_for_part(idx, part) for idx, part in enumerate(state.parts)]


def _preview_record_for_part(index: int, part: PartState) -> dict[str, Any]:
    payload = _preview_payload_for_part(part)
    raw_mins, raw_maxs = _mesh_bbox(payload)
    mins, maxs = _transformed_bbox(raw_mins, raw_maxs, part)
    source_path = _preview_source_path_for_part(part)
    return {
        "index": index,
        "part": part,
        "payload": payload,
        "mins": np.asarray(mins, dtype=float),
        "maxs": np.asarray(maxs, dtype=float),
        "source_path": source_path,
        "mesh_ext": Path(source_path or "").suffix.lower(),
        "display_name": _part_display_name(part),
        "segment": assemble.parse_part_name(part.name)[2],
        "color": list(assemble.pick_color(part.name, index)[1]),
    }


def _preview_part_info(records: list[dict[str, Any]], offsets: list[list[float]]):
    part_info = []
    mesh_payloads = []
    for record, offset in zip(records, offsets):
        part_info.append((
            record["display_name"],
            None,
            assemble.Location(assemble.Vector(*offset)),
            tuple(record["color"]),
            record["segment"],
            record["mesh_ext"] in assemble.MESH_EXTENSIONS,
        ))
        mesh_payloads.append(record["payload"])
    return part_info, mesh_payloads


def _render_preview_assembly_fast(output_path: str, resolution: int = 900, target_triangles: int = 12000) -> None:
    import pyvista as pv

    pv.OFF_SCREEN = True
    records = _preview_records()
    offsets = _preview_stack_offsets(records)
    progress.begin("Rendering", len(records) + 1, "Preparing preview meshes...")
    plotter = pv.Plotter(off_screen=True, window_size=[resolution, resolution])
    plotter.set_background("white", top="lightsteelblue")

    for idx, (record, offset) in enumerate(zip(records, offsets), start=1):
        part = record["part"]
        payload = _preview_thumb_payload_for_part(part, target_triangles=target_triangles)
        vertices, faces = _payload_arrays(payload)
        if len(vertices) == 0 or len(faces) == 0:
            progress.advance(1, f"Skipped {part.name}")
            continue

        transformed = vertices @ _part_linear_transform_matrix(part).T
        transformed = transformed + np.asarray(offset, dtype=float) + np.asarray(part.settle_offset, dtype=float)
        face_data = np.concatenate(
            [np.full((len(faces), 1), 3, dtype=np.int64), np.asarray(faces, dtype=np.int64)],
            axis=1,
        ).reshape(-1)
        mesh = pv.PolyData(np.asarray(transformed, dtype=float), face_data)
        plotter.add_mesh(
            mesh,
            color=tuple(record["color"]),
            ambient=0.28,
            diffuse=0.7,
            specular=0.35,
            specular_power=30,
            smooth_shading=True,
        )
        progress.advance(1, f"Prepared render mesh {idx} of {len(records)}: {part.name}")

    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.camera.zoom(0.92)

    progress.advance(1, "Capturing screenshot...")
    plotter.screenshot(output_path)
    plotter.close()
    progress.finish(f"Render saved: {output_path}")


def _export_preview_assembly_mesh(output_path: str) -> None:
    import trimesh

    records = _preview_records()
    offsets = _preview_stack_offsets(records)
    progress.begin("Exporting", len(records) + 1, "Preparing assembly mesh...")
    meshes = []
    for idx, (record, offset) in enumerate(zip(records, offsets), start=1):
        part = record["part"]
        vertices, faces = _payload_arrays(record["payload"])
        if len(vertices) == 0 or len(faces) == 0:
            progress.advance(1, f"Skipped {part.name}")
            continue
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        mesh.apply_transform(_part_transform_matrix_4x4(part))
        mesh.apply_translation(np.asarray(offset, dtype=float) + np.asarray(part.settle_offset, dtype=float))
        meshes.append(mesh)
        progress.advance(1, f"Prepared assembly mesh {idx} of {len(records)}: {part.name}")

    merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else (meshes[0] if meshes else None)
    if merged is None:
        raise RuntimeError("no mesh geometry available for assembly export")
    progress.advance(1, "Writing assembly mesh...")
    merged.export(output_path)
    progress.finish(f"Assembly exported: {output_path}")


def _build_fast_mesh_scene() -> dict[str, Any]:
    progress.begin("Scene", len(state.parts) * 2, "Stacking preview meshes for scene...")
    records = []
    for idx, part in enumerate(state.parts, start=1):
        with _SlowProgressDetail(
            f"Preparing preview {idx} of {len(state.parts)}: {part.name}",
            lambda elapsed, i=idx, name=part.name:
                f"Reading preview mesh {i} of {len(state.parts)}: {name} ({elapsed:.0f}s)",
        ):
            record = _preview_record_for_part(idx - 1, part)
        records.append(record)
        progress.advance(1, f"Prepared preview {idx} of {len(state.parts)}: {part.name}")

    offsets = _preview_stack_offsets(records)
    combined = []
    part_meshes = []
    for idx, (record, offset) in enumerate(zip(records, offsets), start=1):
        part = record["part"]
        mesh_url = _mesh_file_url(record["source_path"]) if record["mesh_ext"] == ".stl" else None
        use_mesh_url = bool(mesh_url)
        part_meshes.append({
            "name": part.name,
            "filePath": part.file_path,
            "meshSourcePath": record["source_path"],
            "previewPath": part.preview_path,
            "previewOnly": part.preview_only,
            "previewLabel": (
                f"Preview mesh: {Path(part.preview_path).name}"
                if part.preview_path
                else None
            ),
            "orientationSteps": _serialize_orientation_steps(part.orientation_steps),
            "rot": list(part.rot_xyz),
            "scale": part.manual_scale,
            "material": part.material,
            "mesh": None if use_mesh_url else record["payload"],
            "thumbMesh": _preview_thumb_payload_for_part(part),
            "meshUrl": mesh_url,
            "meshFormat": "stl" if use_mesh_url else None,
        })
        settle = np.asarray(part.settle_offset, dtype=float)
        combined.append({
            "name": record["display_name"],
            "color": record["color"],
            "partIndex": idx - 1,
            "offset": (np.asarray(offset, dtype=float) + settle).tolist(),
        })
        progress.advance(1, f"Prepared assembled mesh {idx} of {len(state.parts)}: {part.name}")

    progress.finish("Scene ready")
    _record_debug_log("server", "scene-fast-path", "Used preview mesh fast path", {
        "parts": [part.name for part in state.parts],
        "axis": state.axis,
    })
    _record_debug_log("server", "scene-done", "Scene payload ready", {
        "parts": len(part_meshes),
        "combined": len(combined),
    })
    return {"parts": part_meshes, "combined": combined}


def _json_api_error(message: str, status: int = 400, **extra):
    payload = {"error": message}
    payload.update(extra)
    return jsonify(payload), status


def _load_cached_part(filepath: str, require_solid: bool = False) -> tuple[cq.Workplane, str]:
    path = Path(filepath).resolve()
    key = (str(path), path.stat().st_mtime, require_solid)
    cached = _SHAPE_CACHE.get(key)
    if cached is not None:
        shape, name = cached
        return cq.Workplane("XY").newObject([cq.Shape(shape)]), name

    wp, name = assemble.load_part(str(path), require_solid=require_solid)
    _SHAPE_CACHE[key] = (wp.val().wrapped, name)
    return wp, name


def _find_preview_surrogate(path: Path) -> Path | None:
    if path.suffix.lower() not in assemble.CAD_EXTENSIONS:
        return None
    if path.stat().st_size < _PREVIEW_SURROGATE_MIN_BYTES:
        return None

    candidate_stems = [path.stem]
    for suffix in ("_precut", "_cut", "_assembly"):
        if path.stem.lower().endswith(suffix):
            candidate_stems.append(path.stem[: -len(suffix)])

    mesh_exts = sorted(assemble.MESH_EXTENSIONS)
    for stem in candidate_stems:
        for ext in mesh_exts:
            candidate = path.with_name(f"{stem}{ext}")
            if candidate.exists():
                return candidate
            candidate_upper = path.with_name(f"{stem}{ext.upper()}")
            if candidate_upper.exists():
                return candidate_upper
    return None


def _ensure_real_geometry(part: PartState) -> None:
    if not part.preview_only:
        return
    wp, _name = _load_cached_part(part.file_path, require_solid=False)
    part.shape = wp.val().wrapped
    part.mesh_source_path = part.file_path
    part.preview_only = False
    part.preview_path = None
    part.preview_ext = None


def _ensure_all_real_geometry() -> None:
    for part in state.parts:
        _ensure_real_geometry(part)


def _orientation_axis_index() -> int:
    return {"x": 0, "y": 1, "z": 2}.get(state.axis, 2)


def _orientation_slice_metrics(rotated: np.ndarray) -> tuple[float, float, int, float]:
    axis_index = _orientation_axis_index()
    other_axes = [idx for idx in range(3) if idx != axis_index]
    coords = rotated[:, axis_index]
    span = float(coords.max() - coords.min())
    if span < 1e-6:
        return 1e6, 1e6, 0, 1e6

    mins = rotated.min(axis=0)
    maxs = rotated.max(axis=0)
    radial_extents = [float(maxs[idx] - mins[idx]) for idx in other_axes]
    radial_balance = abs(radial_extents[0] - radial_extents[1]) / max(max(radial_extents), 1e-6)

    edges = np.linspace(float(coords.min()), float(coords.max()), 13)
    center_drifts: list[float] = []
    radius_cvs: list[float] = []
    min_points = max(24, min(200, len(rotated) // 20))
    used = 0
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (coords >= start) & (coords < end)
        pts = rotated[mask][:, other_axes]
        if len(pts) < min_points:
            continue
        center = pts.mean(axis=0)
        radii = np.linalg.norm(pts - center, axis=1)
        mean_radius = float(np.mean(radii))
        if mean_radius < 1e-6:
            continue
        center_drifts.append(float(np.linalg.norm(center)) / mean_radius)
        radius_cvs.append(float(np.std(radii)) / mean_radius)
        used += 1

    if not used:
        return 1e6, 1e6, 0, radial_balance + 1.0
    return float(np.mean(center_drifts)), float(np.mean(radius_cvs)), used, radial_balance


def _cylinder_taper_penalty(rotated: np.ndarray) -> float:
    axis_index = _orientation_axis_index()
    other_axes = [idx for idx in range(3) if idx != axis_index]
    coords = rotated[:, axis_index]
    mid = float((coords.min() + coords.max()) / 2.0)
    bottom = rotated[coords < mid][:, other_axes]
    top = rotated[coords >= mid][:, other_axes]
    if not len(bottom) or not len(top):
        return 0.0
    bottom_center = bottom.mean(axis=0)
    top_center = top.mean(axis=0)
    r_bottom = float(np.mean(np.linalg.norm(bottom - bottom_center, axis=1)))
    r_top = float(np.mean(np.linalg.norm(top - top_center, axis=1)))
    if r_bottom < 1e-6:
        return 0.0
    return max(0.0, (r_top - r_bottom) / r_bottom)


def _axis_alignment_candidates(target: np.ndarray) -> list[tuple[str, list[tuple[tuple[float, float, float], float]]]]:
    axes = [
        ("+x", np.array([1.0, 0.0, 0.0], dtype=float)),
        ("-x", np.array([-1.0, 0.0, 0.0], dtype=float)),
        ("+y", np.array([0.0, 1.0, 0.0], dtype=float)),
        ("-y", np.array([0.0, -1.0, 0.0], dtype=float)),
        ("+z", np.array([0.0, 0.0, 1.0], dtype=float)),
        ("-z", np.array([0.0, 0.0, -1.0], dtype=float)),
    ]
    return [(label, _rotation_steps_to_axis(source_axis, target, False)) for label, source_axis in axes]


def _solve_orientation_steps_axis_aligned(workflow: str) -> list[tuple[tuple[float, float, float], float]]:
    clouds: list[np.ndarray] = []
    progress.begin("Auto-orient", len(state.parts) + 7, "Checking axis-aligned orientations...")
    for idx, part in enumerate(state.parts, start=1):
        payload = _orientation_payload_for_part(part)
        clouds.append(_sample_vertices(payload))
        progress.advance(1, f"Sampled orientation proxy {idx} of {len(state.parts)}: {part.name}")

    if not clouds:
        progress.finish("Auto-orient skipped")
        return []

    all_vertices = np.vstack(clouds)
    target = _axis_vector(state.axis)
    ranked: list[tuple[float, str, list[tuple[tuple[float, float, float], float]]]] = []
    for idx, (label, steps) in enumerate(_axis_alignment_candidates(target), start=1):
        rotated = _apply_steps_to_vertices(all_vertices, steps)
        center_drift, radius_cv, slice_count, radial_balance = _orientation_slice_metrics(rotated)
        taper_penalty = _cylinder_taper_penalty(rotated) if workflow == "cylinder" else 0.0
        coverage_penalty = 0.35 * max(0, 4 - slice_count)
        score = (center_drift * 4.0) + (radius_cv * 8.0) + (radial_balance * 2.0) + coverage_penalty + (taper_penalty * 3.0)
        ranked.append((score, label, steps))
        progress.advance(1, f"Checked axis orientation {idx} of 6: {label} (score {score:.3f})")

    ranked.sort(key=lambda item: item[0])
    best_score, best_label, best_steps = ranked[0]
    progress.note(f"Selected axis orientation {best_label} (score {best_score:.3f})")
    progress.finish("Auto-orient complete")
    return best_steps


def _solve_orientation_steps_fine(workflow: str) -> list[tuple[tuple[float, float, float], float]]:
    clouds: list[np.ndarray] = []
    progress.begin("Auto-orient", len(state.parts) + 2, "Sampling preview meshes for orientation...")
    for idx, part in enumerate(state.parts, start=1):
        payload = _orientation_payload_for_part(part)
        vertices, faces = _payload_arrays(payload)
        if len(vertices):
            try:
                if len(faces):
                    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64), process=False)
                    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=20)
                    vertices = np.asarray(mesh.vertices)
            except Exception:
                pass
            clouds.append(_sample_vertices(_payload_from_arrays(vertices, faces)))
        progress.advance(1, f"Sampled orientation proxy {idx} of {len(state.parts)}: {part.name}")

    if not clouds:
        progress.finish("Auto-orient skipped")
        return []

    all_vertices = np.vstack(clouds)
    principal = assemble.pca_principal_axis(all_vertices) if assemble.gpu_enabled() else None
    if principal is None:
        centered = all_vertices - all_vertices.mean(axis=0)
        cov = (centered.T @ centered) / max(1, len(centered))
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = eigenvectors[:, np.argmax(eigenvalues)]

    target = _axis_vector(state.axis)
    if float(np.dot(principal, target)) < 0:
        principal = -principal

    progress.advance(1, "Solved principal axis from preview meshes")

    steps = _rotation_steps_to_axis(principal, target, False)
    if workflow == "cylinder":
        rotated = _apply_steps_to_vertices(all_vertices, steps)
        axis_index = {"x": 0, "y": 1, "z": 2}.get(state.axis, 2)
        other_axes = [idx for idx in range(3) if idx != axis_index]
        coords = rotated[:, axis_index]
        mid = float((coords.min() + coords.max()) / 2.0)
        bottom = rotated[coords < mid]
        top = rotated[coords >= mid]
        if len(bottom) and len(top):
            r_bottom = float(np.mean(np.linalg.norm(bottom[:, other_axes], axis=1)))
            r_top = float(np.mean(np.linalg.norm(top[:, other_axes], axis=1)))
            if r_top > r_bottom * 1.001:
                steps = _rotation_steps_to_axis(principal, target, True)
    progress.advance(1, "Resolved cylinder orientation")
    progress.finish("Auto-orient complete")
    return steps


def _solve_orientation_steps(workflow: str) -> tuple[list[tuple[tuple[float, float, float], float]], str]:
    if state.fine_orient:
        return _solve_orientation_steps_fine(workflow), "fine-grained preview solve"
    return _solve_orientation_steps_axis_aligned(workflow), "axis-aligned six-way heuristic"


def _centroid_along_stack_axis(part: PartState) -> float:
    mins, maxs = _preview_bounds_for_part(part)
    axis_index = {"x": 0, "y": 1, "z": 2}.get(state.axis, 2)
    return (mins[axis_index] + maxs[axis_index]) / 2.0


def _preview_cross_section_diameter(part: PartState) -> float:
    mins, maxs = _preview_bounds_for_part(part)
    axis_index = {"x": 0, "y": 1, "z": 2}.get(state.axis, 2)
    other_axes = [idx for idx in range(3) if idx != axis_index]
    return max(float(maxs[idx] - mins[idx]) for idx in other_axes)


def _autoscale_preview_parts() -> list[tuple[PartState, float]]:
    by_level: dict[int, dict[str, list[PartState]]] = {}
    for part in state.parts:
        tier, levels, _segment = assemble.parse_part_name(part.name)
        if tier is None or levels is None:
            continue
        for level in levels:
            by_level.setdefault(level, {}).setdefault(tier, []).append(part)

    outer_diams: dict[int, float] = {}
    for level, tier_map in by_level.items():
        outers = tier_map.get("outer")
        if outers:
            outer_diams[level] = _preview_cross_section_diameter(outers[0])

    if outer_diams:
        sorted_values = sorted(outer_diams.values())
        ref_outer = sorted_values[len(sorted_values) // 2]
    else:
        ref_outer = None

    scaled: list[tuple[PartState, float]] = []
    for level, tier_map in by_level.items():
        outer_diam = outer_diams.get(level, ref_outer)
        if outer_diam is None:
            continue
        for tier_name in ("mid", "inner"):
            for part in tier_map.get(tier_name, []):
                current = _preview_cross_section_diameter(part)
                if current <= 1e-9:
                    continue
                target_diam = assemble._parse_target_diameter(part.name)
                factor = None
                if target_diam is not None:
                    factor = target_diam / current
                else:
                    ratio = outer_diam / current
                    if ratio > assemble._INNER_TINY_RATIO:
                        mm_factor = assemble._MM_TO_INCH
                        cm_factor = assemble._CM_TO_INCH
                        mm_diff = abs(outer_diam - (current * mm_factor))
                        cm_diff = abs(outer_diam - (current * cm_factor))
                        factor = mm_factor if mm_diff <= cm_diff else cm_factor
                if factor is not None and abs(factor - 1.0) > 0.01:
                    scaled.append((part, factor))
    return scaled


def _build_scene() -> dict[str, Any]:
    stack_parts = _parts_for_stack()
    if not stack_parts:
        return {"parts": [], "combined": []}
    if _can_use_fast_mesh_scene(state.parts):
        return _build_fast_mesh_scene()
    _record_debug_log("server", "scene-start", "Building scene payload", {
        "parts": [p.name for p in state.parts],
        "count": len(state.parts),
    })
    axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
    _assy, part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
    part_info = _apply_settle_offsets(part_info)
    progress.begin("Scene", len(state.parts) + len(part_info), "Stacking parts for scene...")

    part_meshes = []
    combined = []
    part_index_by_name = {entry[1]: entry[4] for entry in stack_parts}
    for idx, (entry, part_state) in enumerate(zip(stack_parts, state.parts), start=1):
        can_reuse_cached_mesh = bool(part_state.mesh_source_path)
        with _SlowProgressDetail(
            f"Preparing preview {idx} of {len(state.parts)}: {part_state.name}",
            lambda elapsed, i=idx, name=part_state.name:
                f"Tessellating preview {i} of {len(state.parts)}: {name} ({elapsed:.0f}s)",
        ):
            if can_reuse_cached_mesh:
                mesh = _load_mesh_payload_from_cache_file(
                    part_state.mesh_source_path,
                    entry[0].val().wrapped,
                )
            else:
                mesh = _mesh_payload(entry[0].val().wrapped)
        part_meshes.append({
            "name": part_state.name,
            "filePath": part_state.file_path,
            "meshSourcePath": part_state.mesh_source_path,
            "previewPath": part_state.preview_path,
            "previewOnly": part_state.preview_only,
            "previewLabel": (
                f"Preview mesh: {Path(part_state.preview_path).name}"
                if part_state.preview_only and part_state.preview_path
                else None
            ),
            "orientationSteps": _serialize_orientation_steps(part_state.orientation_steps),
            "rot": list(part_state.rot_xyz),
            "scale": part_state.manual_scale,
            "material": part_state.material,
            "mesh": mesh,
            "thumbMesh": (
                _load_decimated_mesh_payload(
                    part_state.mesh_source_path,
                    entry[0].val().wrapped,
                    tolerance=0.8,
                    target_triangles=_THUMB_TRIANGLE_TARGET,
                )
                if can_reuse_cached_mesh
                else _decimate_payload(mesh, _THUMB_TRIANGLE_TARGET)
            ),
        })
        progress.advance(1, f"Prepared preview {idx} of {len(state.parts)}: {part_state.name}")

    for idx, (name, shape, loc, rgb, *_rest) in enumerate(part_info, start=1):
        source_index = part_index_by_name.get(name)
        if source_index is not None:
            settle = np.asarray(state.parts[source_index].settle_offset, dtype=float)
            loc_offset = np.asarray(list(loc.toTuple()[0]), dtype=float)
            combined.append({
                "name": name,
                "color": list(rgb),
                "partIndex": source_index,
                "offset": (loc_offset + settle).tolist(),
            })
        else:
            moved = assemble.apply_location(shape, loc)
            with _SlowProgressDetail(
                f"Building assembled mesh {idx} of {len(part_info)}: {name}",
                lambda elapsed, i=idx, part_name=name:
                    f"Tessellating assembled mesh {i} of {len(part_info)}: {part_name} ({elapsed:.0f}s)",
            ):
                m = _mesh_payload(moved)
            combined.append({"name": name, "color": list(rgb), "mesh": m})
        progress.advance(1, f"Prepared assembled mesh {idx} of {len(part_info)}: {name}")

    progress.finish("Scene ready")
    _record_debug_log("server", "scene-done", "Scene payload ready", {
        "parts": len(part_meshes),
        "combined": len(combined),
    })
    return {"parts": part_meshes, "combined": combined}


# ---------------------------------------------------------------------------
# Gradient module loader
# ---------------------------------------------------------------------------

def _load_gradient_module():
    mod_path = Path(__file__).resolve().parent / "wrl-color-gradient-app" / "wrl_color_gradient.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Gradient module not found: {mod_path}")
    spec = importlib.util.spec_from_file_location("wrl_color_gradient", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import gradient module from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        webui_version=_WEBUI_VERSION,
        workflows=SUPPORTED_WORKFLOWS,
        current_workflow=state.workflow,
        fine_orient=state.fine_orient,
    )


@app.route("/api/progress")
def get_progress():
    """Return current progress state as JSON for polling."""
    return jsonify(progress.snapshot())


@app.route("/api/version")
def get_version():
    return jsonify({"webui_version": _WEBUI_VERSION})


@app.route("/api/debug-log", methods=["GET", "POST", "DELETE"])
def debug_log():
    if request.method == "GET":
        return jsonify({"entries": _debug_log_snapshot()})
    if request.method == "DELETE":
        with _DEBUG_LOG_LOCK:
            _DEBUG_LOG.clear()
        if _DEBUG_LOG_PATH.exists():
            _DEBUG_LOG_PATH.write_text("", encoding="utf-8")
        return jsonify({"ok": True})

    data = request.get_json(force=True)
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        return jsonify({"error": "entries must be a list"}), 400
    accepted = 0
    for raw in entries[-100:]:
        if not isinstance(raw, dict):
            continue
        _record_debug_log(
            "client",
            str(raw.get("kind", "client")),
            str(raw.get("message", ""))[:500],
            {
                "client_ts": raw.get("ts"),
                "meta": raw.get("meta"),
            },
        )
        accepted += 1
    return jsonify({"ok": True, "accepted": accepted})


@app.route("/api/mesh-file")
def mesh_file():
    raw_path = request.args.get("path", "")
    if not raw_path:
        return jsonify({"error": "missing path"}), 400
    try:
        path = _safe_resolve(raw_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 403
    if not path.exists() or not path.is_file():
        return jsonify({"error": f"file not found: {raw_path}"}), 404
    if path.suffix.lower() not in assemble.MESH_EXTENSIONS:
        return jsonify({"error": f"unsupported mesh extension: {path.suffix}"}), 400
    return send_file(path, conditional=True)


@app.route("/api/progress/stream")
def stream_progress():
    """Server-Sent Events stream for real-time progress updates."""
    q: queue.Queue = queue.Queue()

    def listener(stage, current, total, message):
        q.put(progress.snapshot())

    progress.add_listener(listener)

    def generate():
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            progress.remove_listener(listener)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/files")
def list_files():
    out = [p.name for p in sorted(state.parts_dir.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    return jsonify({"files": out, "parts_dir": str(state.parts_dir)})


@app.route("/api/directories")
def list_directories():
    raw = request.args.get("path", "").strip() or "."
    try:
        current_dir = _safe_resolve(raw)
    except ValueError as e:
        return jsonify({"error": str(e)}), 403
    if not current_dir.exists() or not current_dir.is_dir():
        return jsonify({"error": f"directory not found: {current_dir}"}), 404

    cwd = Path.cwd().resolve()
    directories = []
    for child in sorted(current_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        file_count = sum(
            1 for p in child.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
        )
        directories.append({
            "name": child.name,
            "path": _relative_to_cwd(child),
            "file_count": file_count,
        })

    parent = None
    if current_dir != cwd and cwd in current_dir.parents:
        parent = _relative_to_cwd(current_dir.parent)

    return jsonify({
        "root": ".",
        "current": _relative_to_cwd(current_dir),
        "parent": parent,
        "directories": directories,
    })


@app.route("/api/parts-dir", methods=["GET", "PATCH"])
def parts_dir_config():
    if request.method == "GET":
        return jsonify({"parts_dir": str(state.parts_dir)})

    data = request.get_json(force=True)
    raw = str(data.get("path", "")).strip()
    if not raw:
        return jsonify({"error": "missing path"}), 400
    try:
        parts_dir = _safe_resolve(raw)
    except ValueError as e:
        return jsonify({"error": str(e)}), 403
    if not parts_dir.exists() or not parts_dir.is_dir():
        return jsonify({"error": f"directory not found: {parts_dir}"}), 404

    state.parts_dir = parts_dir
    return jsonify({"ok": True, "parts_dir": str(state.parts_dir)})


@app.route("/api/upload-part", methods=["POST"])
def upload_part():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    f = request.files["file"]
    name = Path(f.filename or "").name
    if not name:
        return jsonify({"error": "invalid filename"}), 400
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXT:
        return jsonify({"error": f"unsupported file extension: {ext}"}), 400

    target = state.parts_dir / name
    f.save(target)
    return jsonify({"ok": True, "file": name, "saved_to": str(target)})


@app.route("/api/gradient/files")
def list_gradient_files():
    out = [p.name for p in sorted(state.parts_dir.iterdir()) if p.is_file() and p.suffix.lower() in GRADIENT_EXT]
    return jsonify({"files": out})


@app.route("/api/capability/wrl_gradient", methods=["POST"])
def run_wrl_gradient_capability():
    data = request.get_json(force=True)
    input_file = data.get("input")
    if not input_file:
        return jsonify({"error": "missing input"}), 400

    try:
        input_path = _safe_resolve(input_file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 403
    if not input_path.exists():
        return jsonify({"error": f"input not found: {input_path}"}), 404

    output_ply = data.get("output", "web_colored_output.ply")
    render_svg = data.get("render")
    mode = data.get("mode", "top-bottom")

    module = _load_gradient_module()
    colorizer = module.MeshThermalColorizer(
        source_color=module.MeshThermalColorizer.hex_color(data.get("sourceColor", "#FF0000")),
        sink_color=module.MeshThermalColorizer.hex_color(data.get("sinkColor", "#0000FF")),
        mode=mode,
        source_temp=float(data.get("sourceTemp", 500.0)),
        sink_temp=float(data.get("sinkTemp", 300.0)),
        ambient_temp=float(data.get("ambientTemp", 300.0)),
        material=data.get("material", "stainless_steel"),
        dt=float(data.get("dt", 0.1)),
        max_steps=int(data.get("maxSteps", 4000)),
        diffusion_rate=float(data.get("diffusionRate", 1.0)),
        source_band=float(data.get("sourceBand", 0.03)),
        sink_band=float(data.get("sinkBand", 0.03)),
        radial_inner=float(data.get("radialInner", 0.1)),
        radial_outer=float(data.get("radialOuter", 0.95)),
        palette=(data.get("palette") or None),
        reverse_palette=bool(data.get("reversePalette", False)),
    )
    colorizer.process(str(input_path), output_ply, render_svg)

    return jsonify({
        "ok": True,
        "message": f"Gradient output written to {output_ply}",
        "output": output_ply,
        "render": render_svg,
    })


@app.route("/api/load", methods=["POST"])
def load_parts():
    data = request.get_json(force=True)
    files = data.get("files", [])
    include_scene = bool(data.get("include_scene"))
    if not isinstance(files, list) or not files:
        return _json_api_error("no files selected", 400)

    _record_debug_log("server", "load-start", "Loading parts", {"files": files})
    loaded_parts: list[PartState] = []
    progress.begin("Loading", len(files), "Loading parts...")
    try:
        for index, rel in enumerate(files, start=1):
            path = (state.parts_dir / rel).resolve()
            display_name = path.stem
            if not str(path).startswith(str(state.parts_dir)):
                progress.finish("Load failed")
                return _json_api_error(f"Path escapes parts directory: {rel}", 403, file=rel)
            if not path.exists() or not path.is_file():
                progress.finish("Load failed")
                return _json_api_error(f"file not found: {rel}", 404, file=rel)
            try:
                preview_path = _find_preview_surrogate(path)
                load_path = preview_path or path
                if preview_path is not None:
                    progress.note(
                        f"Using lightweight preview {preview_path.name} for {rel}"
                    )
                with _SlowProgressDetail(
                    f"Loading {index} of {len(files)}: {load_path.name}",
                    lambda elapsed, i=index, file_name=load_path.name:
                        f"Still opening {file_name} ({elapsed:.0f}s elapsed)",
                ):
                    wp, name = _load_cached_part(str(load_path), require_solid=False)
            except Exception as exc:
                app.logger.exception("Failed to load part '%s'", path)
                _record_debug_log("server", "load-error", f"Failed to load {rel}", {"error": str(exc)})
                progress.finish(f"Load failed: {rel}")
                return _json_api_error(
                    f"failed to load '{rel}': {exc}",
                    400,
                    file=rel,
                )

            face_count = _count_faces(wp.val().wrapped)
            progress.note(f"Opened {rel} ({face_count} surfaces)")
            _record_debug_log("server", "load-opened", f"Opened {rel}", {
                "faces": face_count,
                "preview": str(preview_path) if preview_path else None,
            })
            loaded_parts.append(PartState(
                file_path=str(path), name=display_name,
                source_ext=path.suffix.lower(), shape=wp.val().wrapped,
                mesh_source_path=str(load_path),
                preview_path=str(preview_path) if preview_path else None,
                preview_ext=preview_path.suffix.lower() if preview_path else None,
                preview_only=preview_path is not None,
            ))
            progress.advance(1, f"Loaded {display_name} ({index} of {len(files)})")

    finally:
        if progress.snapshot()["stage"] == "Loading":
            progress.finish("Load complete")

    state.parts = loaded_parts
    _record_debug_log("server", "load-done", "Load complete", {"count": len(state.parts)})
    payload: dict[str, Any] = {"ok": True, "count": len(state.parts)}
    if include_scene:
        _record_debug_log("server", "load-scene", "Building scene inline with load", {"count": len(state.parts)})
        payload["scene"] = _build_scene()
    return jsonify(payload)


@app.route("/api/scene")
def get_scene():
    try:
        _record_debug_log("server", "scene-request", "Received scene request")
        return jsonify(_build_scene())
    except Exception as exc:
        app.logger.exception("Failed to build scene")
        _record_debug_log("server", "scene-error", "Failed to build scene", {"error": str(exc)})
        return _json_api_error(f"failed to build scene: {exc}", 500)


@app.route("/api/part/<int:idx>", methods=["PATCH"])
def update_part(idx: int):
    if idx < 0 or idx >= len(state.parts):
        return jsonify({"error": "invalid index"}), 404
    data = request.get_json(force=True)
    p = state.parts[idx]
    if "rotation" in data:
        rot = data["rotation"]
        p.rot_xyz = (
            float(rot.get("x", p.rot_xyz[0])),
            float(rot.get("y", p.rot_xyz[1])),
            float(rot.get("z", p.rot_xyz[2])),
        )
    if "scale" in data:
        p.manual_scale = max(0.01, float(data["scale"]))
    if "material" in data:
        material = str(data["material"]).strip().lower()
        p.material = material or None
    return jsonify({"ok": True})


@app.route("/api/stage/<name>", methods=["POST"])
def run_stage(name: str):
    try:
        if name == "auto_orient":
            steps, mode_label = _solve_orientation_steps(state.workflow)
            for part in state.parts:
                part.orientation_steps = list(steps)
                part.auto_oriented = True
                part.settle_offset = (0.0, 0.0, 0.0)
            return jsonify({"ok": True, "message": f"Auto-orient complete using {mode_label}"})

        if name == "auto_stack":
            progress.begin("Auto-stack", len(state.parts), "Ordering parts for stacking...")
            ranked = []
            for idx, part in enumerate(state.parts, start=1):
                ranked.append((_centroid_along_stack_axis(part), idx, part))
                progress.advance(1, f"Ranked {part.name} for stacking")
            state.parts = [part for _centroid, _idx, part in sorted(ranked, key=lambda item: (item[0], item[1]))]
            for part in state.parts:
                part.settle_offset = (0.0, 0.0, 0.0)
            progress.finish("Auto-stack complete")
            return jsonify({"ok": True, "message": "Auto-stack complete"})

        if name == "auto_scale":
            progress.begin("Auto-scale", len(state.parts), "Scaling parts...")
            scaled_parts = {id(part): factor for part, factor in _autoscale_preview_parts()}
            for part in state.parts:
                factor = scaled_parts.get(id(part))
                if factor is not None:
                    part.manual_scale *= factor
                    part.settle_offset = (0.0, 0.0, 0.0)
                    progress.advance(1, f"Scaled {part.name} by x{factor:.3f}")
                else:
                    progress.advance(1, f"Left {part.name} unchanged")
            progress.finish("Auto-scale complete")
            return jsonify({"ok": True, "message": "Auto-scale complete"})

        if name == "auto_drop":
            axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
            preview_records = _preview_records()
            preview_offsets = _preview_stack_offsets(preview_records)
            base_part_info, mesh_payloads = _preview_part_info(preview_records, preview_offsets)
            local_only, local_metrics = assemble.simulate_physics_contact_fast(
                base_part_info, axis_vec, state.gap, rough_drop=False, debug=False, mesh_payloads=mesh_payloads,
            )
            proxy_then_local, proxy_metrics = assemble.simulate_physics_contact_fast(
                base_part_info, axis_vec, state.gap, rough_drop=True, debug=False, mesh_payloads=mesh_payloads,
            )
            use_proxy = proxy_metrics["elapsed_s"] < local_metrics["elapsed_s"]
            chosen = proxy_then_local if use_proxy else local_only
            chosen_metrics = proxy_metrics if use_proxy else local_metrics
            _store_settle_offsets(base_part_info, chosen)
            message = (
                f"Autodrop complete using {chosen_metrics['method']} "
                f"({chosen_metrics['elapsed_s']:.3f}s; "
                f"local_only={local_metrics['elapsed_s']:.3f}s, "
                f"proxy_then_local={proxy_metrics['elapsed_s']:.3f}s)"
            )
            return jsonify({
                "ok": True,
                "message": message,
                "metrics": {
                    "chosen": chosen_metrics,
                    "local_only": local_metrics,
                    "proxy_then_local": proxy_metrics,
                },
            })

        if name == "cut_inner_from_mid":
            _ensure_all_real_geometry()
            progress.begin("Cutting", 2, "Preparing cut...")
            axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
            _assy, part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
            part_info = _apply_settle_offsets(part_info)
            progress.advance(1, "Cutting inner from mid...")
            out = assemble.mid_cut_parts(part_info, output_dir="parts", clearance=0.02, debug=True, section_number=state.section_number)
            progress.finish(f"Cut complete ({len(out)} parts)")
            return jsonify({"ok": True, "message": f"Cut complete ({len(out)} parts)"})

        if name == "export_parts":
            progress.begin("Exporting", 1, "Exporting parts...")
            _export_parts_fast("parts")
            progress.finish("Parts exported")
            return jsonify({"ok": True, "message": "Exported transformed parts to ./parts"})

        if name == "render_whole":
            out = "web_render.png"
            if _can_use_fast_mesh_scene(state.parts):
                _render_preview_assembly_fast(out, resolution=900)
            else:
                _ensure_all_real_geometry()
                axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
                _assy, part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
                part_info = _apply_settle_offsets(part_info)
                data = [(name, shape, loc, rgb) for name, shape, loc, rgb, *_ in part_info]
                assemble.render_assembly(data, out, resolution=1200)
            return jsonify({"ok": True, "message": f"Rendered {out}"})

        if name == "export_whole":
            if state.parts and all(assemble.is_mesh_file(part.file_path) for part in state.parts):
                out = "web_assembly.stl"
                _export_preview_assembly_mesh(out)
            else:
                _ensure_all_real_geometry()
                progress.begin("Exporting", 1, "Exporting assembly...")
                axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
                assy, _part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
                adjusted_info = _apply_settle_offsets(_part_info)
                assy = assemble.Assembly()
                for entry in adjusted_info:
                    part_name, shape, loc, rgb = entry[:4]
                    wp = cq.Workplane("XY").newObject([cq.Shape(shape)])
                    cq_color = assemble.Color(*rgb) if len(rgb) == 3 else assemble.Color(*rgb[:3])
                    assy.add(wp, name=part_name, loc=loc, color=cq_color)
                out = "web_assembly.step"
                assemble.export_assembly_step(assy, out)
                progress.finish("Assembly exported")
            return jsonify({"ok": True, "message": f"Exported {out}"})

        return _json_api_error(f"unknown stage '{name}'", 400)
    except Exception as exc:
        app.logger.exception("Stage '%s' failed", name)
        progress.finish(f"{name} failed")
        return _json_api_error(f"stage '{name}' failed: {exc}", 500)


@app.route("/api/config", methods=["PATCH"])
def set_config():
    data = request.get_json(force=True)
    if "axis" in data and data["axis"] in assemble.AXIS_MAP:
        state.axis = data["axis"]
    if "workflow" in data and data["workflow"] in SUPPORTED_WORKFLOWS:
        state.workflow = data["workflow"]
    if "fine_orient" in data:
        state.fine_orient = bool(data["fine_orient"])
    if "gap" in data:
        state.gap = float(data["gap"])
    if "cut_angle" in data:
        state.cut_angle = float(data["cut_angle"])
    if "section_number" in data:
        sec = data["section_number"]
        state.section_number = None if sec in (None, "") else int(sec)
    return jsonify({"ok": True})


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception("Unhandled server error")
    if request.path.startswith("/api/"):
        return _json_api_error(f"internal server error: {exc}", 500)
    return ("Internal server error", 500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CAD cutter web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12080)
    parser.add_argument("--no-reload", action="store_true",
                        help="Disable auto-reload on source changes")
    args = parser.parse_args()

    use_reloader = not args.no_reload

    # Try livereload for instant browser refresh on static/template changes.
    # Falls back to Flask's built-in reloader for Python source changes.
    if use_reloader:
        try:
            from livereload import Server
            server = Server(app.wsgi_app)
            server.watch("templates/")
            server.watch("static/")
            server.watch("web_ui.py")
            print(f" * LiveReload server on http://{args.host}:{args.port}")
            print(" * Watching templates/, static/, and web_ui.py for changes")
            server.serve(host=args.host, port=args.port)
        except ImportError:
            print(" * livereload not installed, using Flask debug reloader")
            print(" * Install with: pip install livereload")
            app.run(host=args.host, port=args.port, debug=True)
    else:
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
