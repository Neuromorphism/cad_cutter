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
    rel = _relative_to_cwd(Path(path))
    return f"/api/mesh-file?path={rel}"


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
    if not _part_has_runtime_transform(part) and part.mesh_source_path:
        return _load_decimated_mesh_payload(
            part.mesh_source_path,
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
        if part is None or not part.mesh_source_path or _part_has_runtime_transform(part):
            payloads.append(None)
            continue
        try:
            payloads.append(_load_mesh_payload_from_cache_file(part.mesh_source_path, shape))
        except Exception:
            payloads.append(None)
    return payloads


def _can_use_fast_mesh_scene(part_states: list[PartState]) -> bool:
    if not part_states:
        return False
    for part in part_states:
        if not part.mesh_source_path:
            return False
        if Path(part.mesh_source_path).suffix.lower() not in assemble.MESH_EXTENSIONS:
            return False
        if _part_has_runtime_transform(part):
            return False
        tier, _levels, _segment = assemble.parse_part_name(part.name)
        if tier is not None:
            return False
    return True


def _build_fast_mesh_scene() -> dict[str, Any]:
    axis_index = {"x": 0, "y": 1, "z": 2}.get(state.axis, 2)
    center_axes = [idx for idx in range(3) if idx != axis_index]
    progress.begin("Scene", len(state.parts) * 2, "Stacking mesh previews for scene...")

    part_meshes = []
    offsets: list[list[float]] = []
    cursor = 0.0
    for idx, part in enumerate(state.parts, start=1):
        mesh_ext = Path(part.mesh_source_path or "").suffix.lower()
        use_mesh_url = mesh_ext == ".stl"
        payload = None
        with _SlowProgressDetail(
            f"Preparing preview {idx} of {len(state.parts)}: {part.name}",
            lambda elapsed, i=idx, name=part.name:
                f"Reading mesh preview {i} of {len(state.parts)}: {name} ({elapsed:.0f}s)",
        ):
            payload = None if use_mesh_url else _load_mesh_payload_from_cache_file(part.mesh_source_path, part.shape)
        if payload is None:
            payload = _load_mesh_payload_from_cache_file(part.mesh_source_path, part.shape)
        thumb_payload = None if use_mesh_url else _load_decimated_mesh_payload(
            part.mesh_source_path,
            part.shape,
            tolerance=0.8,
            target_triangles=_THUMB_TRIANGLE_TARGET,
        )
        mins, maxs = _mesh_bbox(payload)
        offset = [0.0, 0.0, 0.0]
        for center_axis in center_axes:
            offset[center_axis] = -((mins[center_axis] + maxs[center_axis]) / 2.0)
        offset[axis_index] = cursor - mins[axis_index]
        cursor += (maxs[axis_index] - mins[axis_index]) + state.gap
        offsets.append(offset)
        part_meshes.append({
            "name": part.name,
            "filePath": part.file_path,
            "meshSourcePath": part.mesh_source_path,
            "previewPath": part.preview_path,
            "previewOnly": part.preview_only,
            "previewLabel": (
                f"Preview mesh: {Path(part.preview_path).name}"
                if part.preview_only and part.preview_path
                else None
            ),
            "rot": list(part.rot_xyz),
            "scale": part.manual_scale,
            "material": part.material,
            "mesh": None if use_mesh_url else payload,
            "thumbMesh": thumb_payload,
            "meshUrl": _mesh_file_url(part.mesh_source_path) if use_mesh_url else None,
            "meshFormat": "stl" if use_mesh_url else None,
        })
        progress.advance(1, f"Prepared preview {idx} of {len(state.parts)}: {part.name}")

    combined = []
    for idx, (part, offset) in enumerate(zip(state.parts, offsets), start=1):
        settle = np.asarray(part.settle_offset, dtype=float)
        combined.append({
            "name": part.name,
            "color": list(assemble.pick_color(part.name, idx - 1)[1]),
            "partIndex": idx - 1,
            "offset": (np.asarray(offset, dtype=float) + settle).tolist(),
        })
        progress.advance(1, f"Prepared assembled mesh {idx} of {len(state.parts)}: {part.name}")

    progress.finish("Scene ready")
    _record_debug_log("server", "scene-fast-path", "Used mesh preview fast path", {
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


def _solve_orientation_steps(workflow: str) -> list[tuple[tuple[float, float, float], float]]:
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


def _centroid_along_stack_axis(part: PartState) -> float:
    shape = _apply_manual_transforms(part)
    xmin, ymin, zmin, xmax, ymax, zmax = assemble.get_tight_bounding_box(shape)
    mins = [xmin, ymin, zmin]
    maxs = [xmax, ymax, zmax]
    axis_index = {"x": 0, "y": 1, "z": 2}.get(state.axis, 2)
    return (mins[axis_index] + maxs[axis_index]) / 2.0


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
        can_reuse_cached_mesh = (
            bool(part_state.mesh_source_path)
            and not _part_has_runtime_transform(part_state)
        )
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
    )


@app.route("/api/progress")
def get_progress():
    """Return current progress state as JSON for polling."""
    return jsonify(progress.snapshot())


@app.route("/api/version")
def get_version():
    return jsonify({"webui_version": _WEBUI_VERSION})


@app.route("/api/debug-log", methods=["GET", "POST"])
def debug_log():
    if request.method == "GET":
        return jsonify({"entries": _debug_log_snapshot()})

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
            steps = _solve_orientation_steps(state.workflow)
            for part in state.parts:
                part.orientation_steps = list(steps)
                part.auto_oriented = bool(steps)
                part.settle_offset = (0.0, 0.0, 0.0)
            return jsonify({"ok": True, "message": "Auto-orient complete using decimated preview meshes"})

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
            _ensure_all_real_geometry()
            progress.begin("Auto-scale", len(state.parts), "Scaling parts...")
            entries = [(cq.Workplane("XY").newObject([cq.Shape(_apply_manual_transforms(p))]), p.name) for p in state.parts]
            scaled = assemble.autoscale_parts(entries)
            for i, (wp, _name) in enumerate(scaled):
                state.parts[i].shape = wp.val().wrapped
                state.parts[i].orientation_steps = []
                state.parts[i].manual_scale = 1.0
                state.parts[i].rot_xyz = (0.0, 0.0, 0.0)
                state.parts[i].settle_offset = (0.0, 0.0, 0.0)
                progress.advance(1, f"Scaled {state.parts[i].name}")
            progress.finish("Auto-scale complete")
            return jsonify({"ok": True, "message": "Auto-scale complete"})

        if name == "auto_drop":
            _ensure_all_real_geometry()
            axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
            _assy, base_part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
            mesh_payloads = _autodrop_mesh_payloads(base_part_info)
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
            _ensure_all_real_geometry()
            progress.begin("Exporting", 1, "Exporting parts...")
            assemble.export_transformed_parts(_parts_for_stack(), "parts")
            progress.finish("Parts exported")
            return jsonify({"ok": True, "message": "Exported transformed parts to ./parts"})

        if name == "render_whole":
            _ensure_all_real_geometry()
            axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
            _assy, part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
            part_info = _apply_settle_offsets(part_info)
            data = [(name, shape, loc, rgb) for name, shape, loc, rgb, *_ in part_info]
            out = "web_render.png"
            assemble.render_assembly(data, out, resolution=1200)
            return jsonify({"ok": True, "message": f"Rendered {out}"})

        if name == "export_whole":
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
