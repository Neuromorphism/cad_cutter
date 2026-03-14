#!/usr/bin/env python3
"""Simple web UI for the CAD cutter pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
from flask import Flask, Response, jsonify, render_template, request
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCP.gp import gp_GTrsf

import assemble
from assemble import progress

SUPPORTED_EXT = assemble.ALL_EXTENSIONS
GRADIENT_EXT = {".wrl", ".vrml", ".stl", ".3mf"}


@dataclass
class PartState:
    file_path: str
    name: str
    source_ext: str
    shape: Any
    material: str | None = None
    auto_oriented: bool = False
    rot_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    manual_scale: float = 1.0


@dataclass
class SessionState:
    parts: list[PartState] = field(default_factory=list)
    gap: float = 0.0
    axis: str = "z"
    cut_angle: float = 90.0
    section_number: int | None = None


state = SessionState()
app = Flask(__name__)


def _mesh_payload(shape, tolerance: float = 0.5) -> dict[str, Any]:
    verts, faces = assemble.tessellate_shape(shape, tolerance=tolerance)
    if len(verts) == 0:
        return {"vertices": [], "indices": []}
    indices = faces[:, 1:].reshape(-1).tolist() if len(faces) else []
    return {"vertices": verts.reshape(-1).tolist(), "indices": indices}


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
    for p in state.parts:
        transformed = _apply_manual_transforms(p)
        wp = cq.Workplane("XY").newObject([cq.Shape(transformed)])
        name = p.name if not p.material else f"{p.name}_{p.material}"
        entries.append((wp, name, p.source_ext, assemble.is_mesh_file(p.file_path)))
    return entries


def _build_scene() -> dict[str, Any]:
    parts_for_stack = _parts_for_stack()
    if not parts_for_stack:
        return {"parts": [], "combined": []}
    axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
    _assy, part_info = assemble.stack_parts(parts_for_stack, axis_vec, state.gap)

    part_meshes = []
    combined = []
    for entry, part_state in zip(parts_for_stack, state.parts):
        mesh = _mesh_payload(entry[0].val().wrapped)
        part_meshes.append({
            "name": part_state.name,
            "filePath": part_state.file_path,
            "rot": list(part_state.rot_xyz),
            "scale": part_state.manual_scale,
            "material": part_state.material,
            "mesh": mesh,
        })

    for name, shape, loc, rgb, *_ in part_info:
        moved = assemble.apply_location(shape, loc)
        m = _mesh_payload(moved)
        combined.append({"name": name, "color": list(rgb), "mesh": m})

    return {"parts": part_meshes, "combined": combined}


@app.route("/")
def index():
    return render_template("index.html")


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


@app.route("/api/progress")
def get_progress():
    """Return current progress state as JSON for polling."""
    return jsonify(progress.snapshot())


@app.route("/api/progress/stream")
def stream_progress():
    """Server-Sent Events stream for real-time progress updates."""
    import json
    import queue
    import threading

    q = queue.Queue()

    def listener(stage, current, total, message):
        q.put({"stage": stage, "current": current, "total": total, "message": message})

    progress.add_listener(listener)

    def generate():
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            progress.remove_listener(listener)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/files")
def list_files():
    cwd = Path.cwd()
    out = []
    for p in sorted(cwd.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            out.append(str(p.name))
    return jsonify({"files": out})


@app.route("/api/gradient/files")
def list_gradient_files():
    cwd = Path.cwd()
    out = []
    for p in sorted(cwd.iterdir()):
        if p.is_file() and p.suffix.lower() in GRADIENT_EXT:
            out.append(str(p.name))
    return jsonify({"files": out})


@app.route("/api/capability/wrl_gradient", methods=["POST"])
def run_wrl_gradient_capability():
    data = request.get_json(force=True)
    input_file = data.get("input")
    if not input_file:
        return jsonify({"error": "missing input"}), 400

    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
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
    state.parts.clear()
    progress.begin("Loading", len(files), "Loading parts...")
    for i, rel in enumerate(files):
        path = Path(rel)
        if not path.is_absolute():
            path = Path.cwd() / path
        wp, name = assemble.load_part(str(path), require_solid=False)
        state.parts.append(PartState(file_path=str(path), name=name, source_ext=path.suffix.lower(), shape=wp.val().wrapped))
        progress.advance(1, f"Loaded {name}")
    progress.finish("Parts loaded")
    return jsonify({"ok": True, "count": len(state.parts)})


@app.route("/api/scene")
def get_scene():
    return jsonify(_build_scene())


@app.route("/api/part/<int:index>", methods=["PATCH"])
def update_part(index: int):
    if index < 0 or index >= len(state.parts):
        return jsonify({"error": "invalid index"}), 404
    data = request.get_json(force=True)
    p = state.parts[index]
    if "rotation" in data:
        rot = data["rotation"]
        p.rot_xyz = (float(rot.get("x", p.rot_xyz[0])), float(rot.get("y", p.rot_xyz[1])), float(rot.get("z", p.rot_xyz[2])))
    if "scale" in data:
        p.manual_scale = max(0.01, float(data["scale"]))
    if "material" in data:
        material = str(data["material"]).strip().lower()
        p.material = material or None
    return jsonify({"ok": True})


@app.route("/api/stage/<name>", methods=["POST"])
def run_stage(name: str):
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
        parts_for_stack = _parts_for_stack()
        axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
        _assy, part_info = assemble.stack_parts(parts_for_stack, axis_vec, state.gap)
        progress.advance(1, "Cutting inner from mid...")
        out = assemble.mid_cut_parts(part_info, output_dir="parts", clearance=0.02, debug=True, section_number=state.section_number)
        progress.finish(f"Cut complete ({len(out)} parts)")
        return jsonify({"ok": True, "message": f"Cut complete ({len(out)} parts)"})

    if name == "export_parts":
        progress.begin("Exporting", 1, "Exporting parts...")
        parts_for_stack = _parts_for_stack()
        assemble.export_transformed_parts(parts_for_stack, "parts")
        progress.finish("Parts exported")
        return jsonify({"ok": True, "message": "Exported transformed parts to ./parts"})

    if name == "render_whole":
        parts_for_stack = _parts_for_stack()
        axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
        _assy, part_info = assemble.stack_parts(parts_for_stack, axis_vec, state.gap)
        data = [(name, shape, loc, rgb) for name, shape, loc, rgb, *_ in part_info]
        out = "web_render.png"
        assemble.render_assembly(data, out, resolution=1200)
        return jsonify({"ok": True, "message": f"Rendered {out}"})

    if name == "export_whole":
        progress.begin("Exporting", 1, "Exporting assembly...")
        parts_for_stack = _parts_for_stack()
        axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
        assy, _part_info = assemble.stack_parts(parts_for_stack, axis_vec, state.gap)
        out = "web_assembly.step"
        assemble.export_assembly_step(assy, out)
        progress.finish("Assembly exported")
        return jsonify({"ok": True, "message": f"Exported {out}"})

    return jsonify({"error": f"unknown stage '{name}'"}), 400


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


def main():
    parser = argparse.ArgumentParser(description="CAD cutter web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12080)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
