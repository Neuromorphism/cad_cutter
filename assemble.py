#!/usr/bin/env python3
"""
CAD Assembly Pipeline — Stack, Cut, and Render

Takes multiple CAD/mesh files, assembles them concentrically, optionally cuts
with a radial wedge cutter, exports a .step assembly file, and renders a
professional-quality .png image.

Part naming convention:
  Parts are named as "<tier>_<level>" where:
    tier  = "inner", "mid", or "outer"  (nesting position)
    level = integer starting at 1       (vertical stack order)

  Example filenames: outer_1.step, mid_1.step, inner_1.step, outer_2.step ...

  - Outer parts stack vertically: outer_1 base at z=0, outer_2 on top, etc.
  - Mid parts are XY-centered within the outer part at the same level.
  - Inner parts are XY-centered within the mid part (or outer) at the same level.
  - Parts without a recognized tier/level are treated as outer and stacked in order.

Supported input formats:
  CAD:  .step / .stp / .iges / .igs / .brep
  Mesh: .stl / .obj / .ply / .3mf

Usage examples:
  # Concentric assembly with cut and render
  python assemble.py outer_1.step mid_1.step inner_1.step outer_2.step \\
      -o assembly.step --cut-angle 90 --render assembly.png

  # Simple vertical stack (parts without tier names)
  python assemble.py plate.step shaft.step flange.step -o assy.step --render assy.png
"""

import sys
import os
import re
import argparse
import math
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# CadQuery / OCP imports
# ---------------------------------------------------------------------------
import cadquery as cq
from cadquery import Assembly, Location, Vector, Color
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.gp import gp_Trsf, gp_Vec, gp_Ax2, gp_Pnt, gp_Dir
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE

# ---------------------------------------------------------------------------
# Axis configuration
# ---------------------------------------------------------------------------
AXIS_MAP = {
    "x": Vector(1, 0, 0),
    "y": Vector(0, 1, 0),
    "z": Vector(0, 0, 1),
}

# ---------------------------------------------------------------------------
# Material library (name-keyword -> RGBA 0-1)
# ---------------------------------------------------------------------------
MATERIAL_COLORS = {
    "copper":    (0.85, 0.45, 0.25, 1.0),
    "cu":        (0.85, 0.45, 0.25, 1.0),
    "steel":     (0.55, 0.56, 0.59, 1.0),
    "iron":      (0.44, 0.44, 0.47, 1.0),
    "aluminum":  (0.77, 0.79, 0.82, 1.0),
    "al":        (0.77, 0.79, 0.82, 1.0),
    "gold":      (0.90, 0.72, 0.15, 1.0),
    "au":        (0.90, 0.72, 0.15, 1.0),
    "brass":     (0.78, 0.62, 0.28, 1.0),
    "bronze":    (0.60, 0.32, 0.12, 1.0),
    "chrome":    (0.85, 0.85, 0.88, 1.0),
    "titanium":  (0.52, 0.50, 0.48, 1.0),
    "wood":      (0.40, 0.26, 0.13, 1.0),
    "oak":       (0.40, 0.26, 0.13, 1.0),
    "pine":      (0.55, 0.42, 0.22, 1.0),
    "rubber":    (0.08, 0.08, 0.08, 1.0),
    "plastic":   (0.18, 0.18, 0.18, 1.0),
    "ceramic":   (0.90, 0.90, 0.90, 1.0),
    "glass":     (0.75, 0.85, 0.95, 0.5),
    "stone":     (0.45, 0.45, 0.47, 1.0),
    "concrete":  (0.52, 0.52, 0.54, 1.0),
    "red":       (0.80, 0.15, 0.15, 1.0),
    "blue":      (0.15, 0.25, 0.80, 1.0),
    "green":     (0.15, 0.65, 0.20, 1.0),
    "black":     (0.05, 0.05, 0.05, 1.0),
    "white":     (0.92, 0.92, 0.92, 1.0),
}

DEFAULT_PALETTE = [
    (0.40, 0.60, 0.85, 1.0),  # steel blue
    (0.85, 0.45, 0.30, 1.0),  # terracotta
    (0.45, 0.78, 0.45, 1.0),  # soft green
    (0.90, 0.75, 0.30, 1.0),  # gold
    (0.65, 0.45, 0.80, 1.0),  # lavender
    (0.35, 0.75, 0.75, 1.0),  # teal
    (0.80, 0.55, 0.65, 1.0),  # rose
    (0.70, 0.70, 0.72, 1.0),  # silver
]


# ===================================================================
# Geometry helpers
# ===================================================================

CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs", ".brep"}
MESH_EXTENSIONS = {".stl", ".obj", ".ply", ".3mf"}


def get_bounding_box(shape):
    """Return (xmin,ymin,zmin, xmax,ymax,zmax) for an OCP shape."""
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    return bbox.Get()


def shape_extent(shape, axis):
    """Return (min, max) along the given axis vector."""
    xn, yn, zn, xx, yx, zx = get_bounding_box(shape)
    if axis.x:
        return xn, xx
    elif axis.y:
        return yn, yx
    else:
        return zn, zx


def translate_shape(shape, dx, dy, dz):
    """Return a translated copy of the OCP shape."""
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(dx, dy, dz))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def apply_location(shape, loc):
    """Apply a CadQuery Location to an OCP shape, returning transformed shape."""
    if loc is None:
        return shape
    cq_shape = cq.Shape(shape)
    moved = cq_shape.moved(loc)
    return moved.wrapped


# ===================================================================
# File loading
# ===================================================================

def load_cad_file(filepath):
    """Load a CAD file and return a CadQuery Workplane wrapper."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".step", ".stp"):
        return cq.importers.importStep(filepath)
    elif ext in (".iges", ".igs"):
        from OCP.IGESControl import IGESControl_Reader
        reader = IGESControl_Reader()
        reader.ReadFile(filepath)
        reader.TransferRoots()
        shape = reader.OneShape()
        return cq.Workplane("XY").newObject([cq.Shape(shape)])
    elif ext == ".brep":
        return cq.importers.importBrep(filepath)
    else:
        raise ValueError(f"Unsupported CAD format: {ext}")


def load_mesh_file(filepath):
    """Load a mesh file and convert to CadQuery shape."""
    from OCP.StlAPI import StlAPI_Reader

    ext = os.path.splitext(filepath)[1].lower()
    tmp_stl = None

    if ext != ".stl":
        # Convert to STL via trimesh
        import trimesh
        mesh = trimesh.load(filepath, force="mesh")
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
        tmp_stl = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        mesh.export(tmp_stl.name)
        tmp_stl.close()
        stl_path = tmp_stl.name
    else:
        stl_path = filepath

    try:
        reader = StlAPI_Reader()
        shape = TopoDS_Shape()
        reader.Read(shape, stl_path)
        return cq.Workplane("XY").newObject([cq.Shape(shape)])
    finally:
        if tmp_stl:
            os.unlink(tmp_stl.name)


def load_part(filepath):
    """Load any supported CAD or mesh file, return (cq.Workplane, name)."""
    ext = os.path.splitext(filepath)[1].lower()
    name = os.path.splitext(os.path.basename(filepath))[0]

    if ext in CAD_EXTENSIONS:
        wp = load_cad_file(filepath)
    elif ext in MESH_EXTENSIONS:
        wp = load_mesh_file(filepath)
    else:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: "
            f"{sorted(CAD_EXTENSIONS | MESH_EXTENSIONS)}"
        )
    return wp, name


# ===================================================================
# Color assignment
# ===================================================================

def pick_color(name, index):
    """Pick a color based on part name keywords, or from the palette.
    Returns (cq_color, rgb_tuple) to preserve original sRGB for rendering."""
    lower = name.lower()
    for keyword, rgba in MATERIAL_COLORS.items():
        if keyword in lower:
            rgb = rgba[:3]
            return Color(*rgb), rgb
    c = DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)]
    rgb = c[:3]
    return Color(*rgb), rgb


# ===================================================================
# Part name parsing
# ===================================================================

TIER_ORDER = {"outer": 0, "mid": 1, "inner": 2}
TIER_PATTERN = re.compile(
    r"(?P<tier>inner|mid|outer)[_\- ]?(?P<level>\d+)", re.IGNORECASE
)


def parse_part_name(name):
    """Parse a part name into (tier, level).

    Returns:
        (tier, level): tier is "outer"/"mid"/"inner", level is int >= 1.
        If the name doesn't match, returns (None, None).
    """
    m = TIER_PATTERN.search(name)
    if m:
        return m.group("tier").lower(), int(m.group("level"))
    return None, None


# ===================================================================
# Core assembly (concentric stacking)
# ===================================================================

def stack_parts(parts, axis_vec, gap):
    """
    Assemble parts concentrically based on their tier/level naming.

    Outer parts stack vertically (level 1 at z=0, level 2 on top, etc.).
    Mid parts are XY-centered within the outer part at the same level.
    Inner parts are XY-centered within the mid part (or outer) at the same level.
    Parts without a tier/level name are treated as sequential outer parts.

    Returns a CadQuery Assembly and a list of (name, ocp_shape, location, rgb_tuple).
    """
    # --- Classify parts by tier and level ---
    classified = []  # (tier, level, wp, name, original_index)
    auto_level = 1

    for i, (wp, name) in enumerate(parts):
        tier, level = parse_part_name(name)
        if tier is None:
            # Unrecognized naming — treat as sequential outer parts
            tier = "outer"
            level = auto_level
            auto_level += 1
        classified.append((tier, level, wp, name, i))

    # --- Build the set of levels and group by (level, tier) ---
    levels_map = {}  # level -> {tier -> (wp, name, original_index)}
    for tier, level, wp, name, idx in classified:
        levels_map.setdefault(level, {})[tier] = (wp, name, idx)

    sorted_levels = sorted(levels_map.keys())

    # --- Stack outer parts vertically, center mid/inner within them ---
    assy = Assembly()
    part_info = []
    z_cursor = 0.0  # current Z position for stacking (always Z-axis for vertical)

    for level in sorted_levels:
        tier_group = levels_map[level]

        # Get the outer part for this level (required for positioning)
        if "outer" in tier_group:
            outer_wp, outer_name, outer_idx = tier_group["outer"]
        elif "mid" in tier_group:
            # No outer at this level — use mid as the stacking reference
            outer_wp, outer_name, outer_idx = tier_group["mid"]
        elif "inner" in tier_group:
            outer_wp, outer_name, outer_idx = tier_group["inner"]
        else:
            continue

        outer_shape = outer_wp.val().wrapped
        oxn, oyn, ozn, oxx, oyx, ozx = get_bounding_box(outer_shape)
        outer_height = ozx - ozn
        outer_cx = (oxn + oxx) / 2.0
        outer_cy = (oyn + oyx) / 2.0

        # Place the outer part: shift so its Z-bottom sits at z_cursor,
        # and keep its XY position (centered at origin).
        outer_dz = z_cursor - ozn

        # Process each tier at this level
        for tier in ("outer", "mid", "inner"):
            if tier not in tier_group:
                continue
            wp, name, idx = tier_group[tier]
            shape = wp.val().wrapped
            pxn, pyn, pzn, pxx, pyx, pzx = get_bounding_box(shape)
            part_cx = (pxn + pxx) / 2.0
            part_cy = (pyn + pyx) / 2.0

            if tier == "outer":
                # Outer: just shift Z so base sits at z_cursor
                dx = -part_cx  # center XY at origin
                dy = -part_cy
                dz = z_cursor - pzn
            else:
                # Mid/inner: center XY within the container at this level
                # Find the container (outer for mid, mid for inner)
                if tier == "mid":
                    container_tier = "outer"
                else:
                    container_tier = "mid" if "mid" in tier_group else "outer"

                if container_tier in tier_group:
                    c_wp = tier_group[container_tier][0]
                    c_shape = c_wp.val().wrapped
                    cxn, cyn, czn, cxx, cyx, czx = get_bounding_box(c_shape)
                    # Container center in XY (after it's been moved to origin)
                    cont_cx = 0.0  # container is centered at origin
                    cont_cy = 0.0
                else:
                    cont_cx = 0.0
                    cont_cy = 0.0

                # Center this part at origin XY, base at z_cursor
                dx = cont_cx - part_cx
                dy = cont_cy - part_cy
                dz = z_cursor - pzn

            loc = Location(Vector(dx, dy, dz))
            cq_color, rgb = pick_color(name, idx)

            assy.add(wp, name=name, loc=loc, color=cq_color)
            part_info.append((name, shape, loc, rgb))

        # Advance the Z cursor past the outer part (+ gap)
        z_cursor += outer_height + gap

    return assy, part_info


# ===================================================================
# Cutting
# ===================================================================

def make_cutter(bbox_vals, angle, axis_vec):
    """
    Build a cylindrical-sector cutter from the global bounding box.
    The cutter spans `angle` degrees as a wedge, oriented along the stacking axis.
    """
    xn, yn, zn, xx, yx, zx = bbox_vals

    if axis_vec.y:
        radius = max(abs(xn), abs(xx), abs(zn), abs(zx)) * 2.5
        height = (yx - yn) * 3.0
        start = yn - (yx - yn)
        origin = gp_Pnt(0, start, 0)
        direction = gp_Dir(0, 1, 0)
    elif axis_vec.z:
        radius = max(abs(xn), abs(xx), abs(yn), abs(yx)) * 2.5
        height = (zx - zn) * 3.0
        start = zn - (zx - zn)
        origin = gp_Pnt(0, 0, start)
        direction = gp_Dir(0, 0, 1)
    else:
        radius = max(abs(yn), abs(yx), abs(zn), abs(zx)) * 2.5
        height = (xx - xn) * 3.0
        start = xn - (xx - xn)
        origin = gp_Pnt(start, 0, 0)
        direction = gp_Dir(1, 0, 0)

    radius = max(radius, 1.0)
    height = max(height, 1.0)

    ax = gp_Ax2(origin, direction)
    return BRepPrimAPI_MakeCylinder(ax, radius, height, math.radians(angle)).Shape()


def cut_assembly(assy_compound, cutter_shape):
    """Boolean-cut the assembly compound with the cutter."""
    op = BRepAlgoAPI_Cut(assy_compound, cutter_shape)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("Boolean cut operation failed")
    return op.Shape()


# ===================================================================
# STEP export
# ===================================================================

def export_assembly_step(assy, filepath):
    """Export a CadQuery Assembly to a STEP file."""
    assy.save(filepath, "STEP")


def export_shape_step(shape, filepath):
    """Export a raw OCP shape to STEP."""
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filepath)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed with status {status}")


# ===================================================================
# Tessellation
# ===================================================================

def tessellate_shape(shape, tolerance=0.1, angular=0.5):
    """Tessellate an OCP shape and return (vertices, faces) as numpy arrays."""
    BRepMesh_IncrementalMesh(shape, tolerance, False, angular, True)

    all_verts = []
    all_faces = []
    vert_offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            explorer.Next()
            continue

        trsf = loc.Transformation()
        n_verts = tri.NbNodes()
        n_tris = tri.NbTriangles()

        for i in range(1, n_verts + 1):
            pt = tri.Node(i).Transformed(trsf)
            all_verts.append([pt.X(), pt.Y(), pt.Z()])

        for i in range(1, n_tris + 1):
            t = tri.Triangle(i)
            i1, i2, i3 = t.Get()
            all_faces.append([3,
                              i1 - 1 + vert_offset,
                              i2 - 1 + vert_offset,
                              i3 - 1 + vert_offset])

        vert_offset += n_verts
        explorer.Next()

    if not all_verts:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=int)

    return np.array(all_verts, dtype=float), np.array(all_faces, dtype=int)


# ===================================================================
# Rendering
# ===================================================================

def render_assembly(parts_data, output_path, resolution=2048, cut_shape=None):
    """
    Render the assembly to a PNG using PyVista with professional lighting.

    parts_data: list of (name, ocp_shape, location, color) tuples
    cut_shape:  if set, render this single post-cut shape instead
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    plotter = pv.Plotter(off_screen=True, window_size=[resolution, resolution])

    if cut_shape is not None:
        verts, faces = tessellate_shape(cut_shape, tolerance=0.05)
        if len(verts) > 0:
            mesh = pv.PolyData(verts, faces.ravel())
            plotter.add_mesh(
                mesh, color=(0.65, 0.68, 0.72),
                ambient=0.25, diffuse=0.7, specular=0.5, specular_power=40,
                smooth_shading=True, split_sharp_edges=True,
            )
    else:
        for i, (name, shape, loc, rgb) in enumerate(parts_data):
            translated = apply_location(shape, loc)
            verts, faces = tessellate_shape(translated, tolerance=0.05)
            if len(verts) == 0:
                continue

            mesh = pv.PolyData(verts, faces.ravel())

            plotter.add_mesh(
                mesh, color=rgb,
                ambient=0.25, diffuse=0.7, specular=0.5, specular_power=40,
                smooth_shading=True, split_sharp_edges=True,
            )

    # --- Position lights relative to model bounds ---
    plotter.camera_position = 'iso'
    plotter.reset_camera()
    bounds = plotter.bounds
    if bounds and bounds[0] != bounds[1]:
        cx = (bounds[0] + bounds[1]) / 2
        cy = (bounds[2] + bounds[3]) / 2
        cz = (bounds[4] + bounds[5]) / 2
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        d = span * 2.5  # light distance
    else:
        cx, cy, cz, span, d = 0, 0, 0, 10, 25

    plotter.remove_all_lights()

    # Key light (warm, upper-right-front)
    key = pv.Light(
        position=(cx + d, cy - d, cz + d * 1.2),
        focal_point=(cx, cy, cz), intensity=1.0,
    )
    key.positional = True
    key.cone_angle = 70
    key.diffuse_color = (1.0, 0.98, 0.94)
    plotter.add_light(key)

    # Fill light (cool, upper-left)
    fill = pv.Light(
        position=(cx - d, cy + d * 0.8, cz + d * 0.8),
        focal_point=(cx, cy, cz), intensity=0.6,
    )
    fill.positional = True
    fill.cone_angle = 80
    fill.diffuse_color = (0.90, 0.93, 1.0)
    plotter.add_light(fill)

    # Rim / back light
    rim = pv.Light(
        position=(cx - d * 0.4, cy - d * 1.2, cz + d * 0.6),
        focal_point=(cx, cy, cz), intensity=0.5,
    )
    rim.positional = True
    rim.cone_angle = 60
    plotter.add_light(rim)

    # Top-down fill
    top = pv.Light(
        position=(cx, cy, cz + d * 2),
        focal_point=(cx, cy, cz), intensity=0.4,
    )
    top.positional = True
    top.cone_angle = 90
    plotter.add_light(top)

    # Ambient (strong for overall brightness)
    ambient = pv.Light(intensity=0.45)
    ambient.positional = False
    plotter.add_light(ambient)

    # --- Background gradient ---
    plotter.set_background("white", top="lightsteelblue")

    # --- Floor plane (subtle shadow catcher) ---
    if span > 0:
        floor_z = bounds[4] - span * 0.005
        floor = pv.Plane(
            center=(cx, cy, floor_z), direction=(0, 0, 1),
            i_size=span * 4, j_size=span * 4,
        )
        plotter.add_mesh(floor, color=(0.95, 0.95, 0.95),
                         ambient=0.5, diffuse=0.5, specular=0.0)

    # Camera
    plotter.camera.zoom(1.4)

    # Ambient occlusion
    try:
        plotter.enable_ssao(radius=span * 0.3, bias=0.02, kernel_size=64)
    except Exception:
        pass

    # Anti-aliasing
    try:
        plotter.enable_anti_aliasing('ssaa')
    except Exception:
        pass

    plotter.screenshot(output_path)
    plotter.close()
    print(f"Render saved: {output_path}")


# ===================================================================
# Main pipeline
# ===================================================================

def run_pipeline(args):
    """Execute the full assembly pipeline."""
    # 1. Load all parts
    print(f"Loading {len(args.inputs)} part(s)...")
    parts = []
    for filepath in args.inputs:
        if not os.path.exists(filepath):
            print(f"  WARNING: '{filepath}' not found, skipping.")
            continue
        try:
            wp, name = load_part(filepath)
            parts.append((wp, name))
            print(f"  Loaded: {name}")
        except Exception as e:
            print(f"  ERROR loading '{filepath}': {e}")

    if not parts:
        print("No valid parts loaded. Exiting.")
        return 1

    # 2. Stack parts
    axis_vec = AXIS_MAP[args.axis]
    print(f"\nStacking {len(parts)} part(s) along {args.axis.upper()}-axis (gap={args.gap})...")
    assy, part_info = stack_parts(parts, axis_vec, args.gap)
    print(f"  Assembly created with {len(part_info)} part(s).")

    # 3. Cut (optional) and export STEP
    output_step = args.output
    cut_result_shape = None

    if args.cut_angle is not None:
        # Export pre-cut assembly
        precut_path = os.path.splitext(output_step)[0] + "_precut.step"
        print(f"\nExporting pre-cut assembly: {precut_path}")
        try:
            export_assembly_step(assy, precut_path)
        except Exception as e:
            print(f"  Warning: Assembly export issue: {e}")

        # Build compound for cutting
        print(f"\nCutting assembly at {args.cut_angle} degrees...")
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for name, shape, loc, color in part_info:
            moved = apply_location(shape, loc)
            builder.Add(compound, moved)

        # Bounding box
        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()

        cutter = make_cutter(bbox_vals, args.cut_angle, axis_vec)
        cut_result_shape = cut_assembly(compound, cutter)
        print("  Cut complete.")

        print(f"Exporting cut assembly: {output_step}")
        export_shape_step(cut_result_shape, output_step)
    else:
        print(f"\nExporting assembly: {output_step}")
        try:
            export_assembly_step(assy, output_step)
        except Exception as e:
            print(f"  Assembly export fallback: {e}")
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            for name, shape, loc, color in part_info:
                moved = apply_location(shape, loc)
                builder.Add(compound, moved)
            export_shape_step(compound, output_step)

    print(f"  STEP saved: {output_step}")

    # 4. Render
    if args.render:
        print(f"\nRendering to {args.render} ({args.resolution}x{args.resolution})...")
        render_assembly(
            part_info, args.render,
            resolution=args.resolution,
            cut_shape=cut_result_shape,
        )

    # 5. Summary
    print("\n=== Pipeline Complete ===")
    print(f"  Parts:      {len(parts)}")
    print(f"  Axis:       {args.axis.upper()}")
    print(f"  Gap:        {args.gap}")
    if args.cut_angle is not None:
        print(f"  Cut angle:  {args.cut_angle} deg")
    print(f"  STEP out:   {output_step}")
    if args.render:
        print(f"  Render out: {args.render}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CAD Assembly Pipeline: Stack, Cut, and Render",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Input CAD/mesh files (.step, .stp, .stl, .obj, .ply, .3mf, .iges, .brep)",
    )
    parser.add_argument(
        "-o", "--output", default="assembly.step",
        help="Output STEP file path (default: assembly.step)",
    )
    parser.add_argument(
        "--axis", choices=["x", "y", "z"], default="z",
        help="Stacking axis (default: z)",
    )
    parser.add_argument(
        "--gap", type=float, default=0.0,
        help="Gap between stacked parts in model units (default: 0)",
    )
    parser.add_argument(
        "--cut-angle", type=float, default=None,
        help="Radial cut angle in degrees (e.g. 90, 180). Omit to skip cutting.",
    )
    parser.add_argument(
        "--render", type=str, default=None,
        help="Output PNG render path. Omit to skip rendering.",
    )
    parser.add_argument(
        "--resolution", type=int, default=2048,
        help="Render resolution in pixels (default: 2048)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output",
    )
    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
