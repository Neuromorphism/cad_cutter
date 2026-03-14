#!/usr/bin/env python3
"""Simple web UI for the CAD cutter pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cadquery as cq
from flask import Flask, jsonify, render_template, request
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCP.gp import gp_GTrsf

import assemble

SUPPORTED_EXT = assemble.ALL_EXTENSIONS


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


@app.route("/api/files")
def list_files():
    cwd = Path.cwd()
    out = []
    for p in sorted(cwd.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            out.append(str(p.name))
    return jsonify({"files": out})


@app.route("/api/load", methods=["POST"])
def load_parts():
    data = request.get_json(force=True)
    files = data.get("files", [])
    state.parts.clear()
    for rel in files:
        path = Path(rel)
        if not path.is_absolute():
            path = Path.cwd() / path
        wp, name = assemble.load_part(str(path), require_solid=False)
        state.parts.append(PartState(file_path=str(path), name=name, source_ext=path.suffix.lower(), shape=wp.val().wrapped))
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
        entries = [(cq.Workplane("XY").newObject([cq.Shape(p.shape)]), p.name) for p in state.parts]
        oriented = assemble.orient_to_cylinder(entries, gap=0.0)
        for i, (wp, _name) in enumerate(oriented):
            state.parts[i].shape = wp.val().wrapped
            state.parts[i].rot_xyz = (0.0, 0.0, 0.0)
            state.parts[i].auto_oriented = True
        return jsonify({"ok": True, "message": "Auto-orient complete"})

    if name == "auto_scale":
        entries = [(cq.Workplane("XY").newObject([cq.Shape(p.shape)]), p.name) for p in state.parts]
        scaled = assemble.autoscale_parts(entries)
        for i, (wp, _name) in enumerate(scaled):
            state.parts[i].shape = wp.val().wrapped
            state.parts[i].manual_scale = 1.0
        return jsonify({"ok": True, "message": "Auto-scale complete"})

    if name == "cut_inner_from_mid":
        parts_for_stack = _parts_for_stack()
        axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
        _assy, part_info = assemble.stack_parts(parts_for_stack, axis_vec, state.gap)
        out = assemble.mid_cut_parts(part_info, output_dir="parts", clearance=0.02, debug=True, section_number=state.section_number)
        return jsonify({"ok": True, "message": f"Cut complete ({len(out)} parts)"})

    if name == "export_parts":
        parts_for_stack = _parts_for_stack()
        assemble.export_transformed_parts(parts_for_stack, "parts")
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
        parts_for_stack = _parts_for_stack()
        axis_vec = assemble.AXIS_MAP.get(state.axis, assemble.AXIS_MAP["z"])
        assy, _part_info = assemble.stack_parts(parts_for_stack, axis_vec, state.gap)
        out = "web_assembly.step"
        assemble.export_assembly_step(assy, out)
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
