#!/usr/bin/env python3
"""Simple web UI for the CAD cutter pipeline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
import trimesh
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.gp import gp_GTrsf

import assemble
from assemble import progress

SUPPORTED_EXT = assemble.ALL_EXTENSIONS
GRADIENT_EXT = {".wrl", ".vrml", ".stl", ".3mf"}


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
    rot_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    manual_scale: float = 1.0


@dataclass
class SessionState:
    parts: list[PartState] = field(default_factory=list)
    parts_dir: Path = field(default_factory=lambda: Path.cwd().resolve())
    gap: float = 0.0
    axis: str = "z"
    cut_angle: float = 90.0
    section_number: int | None = None


state = SessionState()
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

_SHAPE_CACHE: dict[tuple[str, float, bool], tuple[Any, str]] = {}
_MESH_PAYLOAD_CACHE: dict[tuple[str, float, int, float], dict[str, Any]] = {}
_PREVIEW_SURROGATE_MIN_BYTES = 25 * 1024 * 1024
_WEBUI_CACHE_DIR = Path(".webui_cache")


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


def _scale_shape(shape, factor: float):
    g = gp_GTrsf()
    g.SetValue(1, 1, factor)
    g.SetValue(2, 2, factor)
    g.SetValue(3, 3, factor)
    return BRepBuilderAPI_GTransform(shape, g, True).Shape()


def _apply_manual_transforms(part: PartState):
    shape = part.shape
    rx, ry, rz = part.rot_xyz
    shape = _rotate_shape(shape, rx, ry, rz)
    if abs(part.manual_scale - 1.0) > 1e-6:
        shape = _scale_shape(shape, part.manual_scale)
    return shape


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


def _build_scene() -> dict[str, Any]:
    stack_parts = _parts_for_stack()
    if not stack_parts:
        return {"parts": [], "combined": []}
    axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
    _assy, part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
    progress.begin("Scene", len(state.parts) + len(part_info), "Stacking parts for scene...")

    part_meshes = []
    combined = []
    part_index_by_name = {entry[1]: entry[4] for entry in stack_parts}
    for idx, (entry, part_state) in enumerate(zip(stack_parts, state.parts), start=1):
        can_reuse_cached_mesh = (
            bool(part_state.mesh_source_path)
            and not part_state.auto_oriented
            and part_state.rot_xyz == (0.0, 0.0, 0.0)
            and abs(part_state.manual_scale - 1.0) < 1e-6
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
        })
        progress.advance(1, f"Prepared preview {idx} of {len(state.parts)}: {part_state.name}")

    for idx, (name, shape, loc, rgb, *_rest) in enumerate(part_info, start=1):
        source_index = part_index_by_name.get(name)
        if source_index is not None:
            combined.append({
                "name": name,
                "color": list(rgb),
                "partIndex": source_index,
                "offset": list(loc.toTuple()[0]),
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
    return render_template("index.html")


@app.route("/api/progress")
def get_progress():
    """Return current progress state as JSON for polling."""
    return jsonify(progress.snapshot())


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
    if not isinstance(files, list) or not files:
        return _json_api_error("no files selected", 400)

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
                progress.finish(f"Load failed: {rel}")
                return _json_api_error(
                    f"failed to load '{rel}': {exc}",
                    400,
                    file=rel,
                )

            face_count = _count_faces(wp.val().wrapped)
            progress.note(f"Opened {rel} ({face_count} surfaces)")
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
    return jsonify({"ok": True, "count": len(state.parts)})


@app.route("/api/scene")
def get_scene():
    try:
        return jsonify(_build_scene())
    except Exception as exc:
        app.logger.exception("Failed to build scene")
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
        _ensure_all_real_geometry()
        if name == "auto_orient":
            progress.begin("Auto-orient", len(state.parts), "Orienting parts...")
            entries = [(cq.Workplane("XY").newObject([cq.Shape(p.shape)]), p.name) for p in state.parts]
            oriented = assemble.orient_to_cylinder(entries, gap=0.0)
            for i, (wp, _name) in enumerate(oriented):
                state.parts[i].shape = wp.val().wrapped
                state.parts[i].rot_xyz = (0.0, 0.0, 0.0)
                state.parts[i].auto_oriented = True
                progress.advance(1, f"Oriented {state.parts[i].name}")
            progress.finish("Auto-orient complete")
            return jsonify({"ok": True, "message": "Auto-orient complete"})

        if name == "auto_scale":
            progress.begin("Auto-scale", len(state.parts), "Scaling parts...")
            entries = [(cq.Workplane("XY").newObject([cq.Shape(p.shape)]), p.name) for p in state.parts]
            scaled = assemble.autoscale_parts(entries)
            for i, (wp, _name) in enumerate(scaled):
                state.parts[i].shape = wp.val().wrapped
                state.parts[i].manual_scale = 1.0
                progress.advance(1, f"Scaled {state.parts[i].name}")
            progress.finish("Auto-scale complete")
            return jsonify({"ok": True, "message": "Auto-scale complete"})

        if name == "cut_inner_from_mid":
            progress.begin("Cutting", 2, "Preparing cut...")
            axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
            _assy, part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
            progress.advance(1, "Cutting inner from mid...")
            out = assemble.mid_cut_parts(part_info, output_dir="parts", clearance=0.02, debug=True, section_number=state.section_number)
            progress.finish(f"Cut complete ({len(out)} parts)")
            return jsonify({"ok": True, "message": f"Cut complete ({len(out)} parts)"})

        if name == "export_parts":
            progress.begin("Exporting", 1, "Exporting parts...")
            assemble.export_transformed_parts(_parts_for_stack(), "parts")
            progress.finish("Parts exported")
            return jsonify({"ok": True, "message": "Exported transformed parts to ./parts"})

        if name == "render_whole":
            axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
            _assy, part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
            data = [(name, shape, loc, rgb) for name, shape, loc, rgb, *_ in part_info]
            out = "web_render.png"
            assemble.render_assembly(data, out, resolution=1200)
            return jsonify({"ok": True, "message": f"Rendered {out}"})

        if name == "export_whole":
            progress.begin("Exporting", 1, "Exporting assembly...")
            axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
            assy, _part_info = assemble.stack_parts(_parts_for_assembly(), axis_vec, state.gap)
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
