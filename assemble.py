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
import glob
import argparse
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from OCP.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Ax2, gp_Pnt, gp_Dir
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
ALL_EXTENSIONS = CAD_EXTENSIONS | MESH_EXTENSIONS


def expand_inputs(raw_inputs):
    """Expand glob patterns and filter to recognized CAD/mesh files.

    Handles three cases:
    1. An existing file path   → kept as-is
    2. A glob pattern (*/?)    → expanded (case-insensitive on all platforms)
    3. A non-existing literal  → kept so the caller can warn about it

    Returns a deduplicated list of paths in stable order.
    """
    result = []
    seen = set()

    for entry in raw_inputs:
        if os.path.exists(entry):
            # Literal file that exists — keep it
            real = os.path.realpath(entry)
            if real not in seen:
                seen.add(real)
                result.append(entry)
        elif any(c in entry for c in ("*", "?", "[", "]")):
            # Glob pattern — expand with case-insensitive matching so
            # *.STEP finds .step/.Step/.STEP and vice versa.
            ci_pattern = _case_insensitive_glob(entry)
            matches = glob.glob(ci_pattern)
            if not matches:
                # Fallback to the original pattern (e.g. already has [])
                matches = glob.glob(entry)
            matches.sort()
            for m in matches:
                ext = os.path.splitext(m)[1].lower()
                if ext in ALL_EXTENSIONS:
                    real = os.path.realpath(m)
                    if real not in seen:
                        seen.add(real)
                        result.append(m)
        else:
            # Literal path that doesn't exist — pass through so caller warns
            result.append(entry)

    return result


def _case_insensitive_glob(pattern):
    """Convert a glob pattern to a case-insensitive version.

    *.STEP  →  *.[sS][tT][eE][pP]
    Works character-by-character on the extension portion.
    """
    parts = []
    for ch in pattern:
        if ch.isalpha():
            parts.append(f"[{ch.lower()}{ch.upper()}]")
        else:
            parts.append(ch)
    return "".join(parts)


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
# Cylinder orientation
# ===================================================================

def orient_to_cylinder(parts):
    """Rotate all parts so the cylinder height axis aligns with Z.

    Uses PCA on the combined vertex cloud of all parts to find the principal
    axis (direction of maximum variance), which corresponds to the cylinder
    height direction.  All parts are rotated together by the same transform so
    that axis → +Z.

    Returns a list of (cq.Workplane, name) tuples with rotated shapes.
    """
    # Collect all vertices via coarse tessellation (speed over accuracy)
    all_verts = []
    for wp, name in parts:
        shape = wp.val().wrapped
        verts, _ = tessellate_shape(shape, tolerance=1.0)
        if len(verts) > 0:
            all_verts.append(verts)

    if not all_verts:
        return parts

    vertices = np.vstack(all_verts)
    center = vertices.mean(axis=0)
    centered = vertices - center

    # 3×3 covariance matrix; eigenvector with largest eigenvalue = principal axis
    cov = (centered.T @ centered) / len(vertices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Largest eigenvalue index → cylinder height axis
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # Prefer orientation with positive Z component for consistency
    if principal[2] < 0:
        principal = -principal

    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(principal, z), -1.0, 1.0))

    px, py, pz = principal
    print(f"  Cylinder axis detected: ({px:.3f}, {py:.3f}, {pz:.3f})")

    # Already aligned with +Z — nothing to do
    if abs(dot) > 1.0 - 1e-6:
        print("  Parts already aligned with Z-axis.")
        return parts

    # Rotation axis = cross(principal, z), then normalize
    rot_axis = np.cross(principal, z)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    angle = math.acos(dot)

    print(f"  Rotating {math.degrees(angle):.1f}° to align cylinder axis with Z.")

    trsf = gp_Trsf()
    ax1 = gp_Ax1(
        gp_Pnt(0.0, 0.0, 0.0),
        gp_Dir(float(rot_axis[0]), float(rot_axis[1]), float(rot_axis[2])),
    )
    trsf.SetRotation(ax1, angle)

    rotated_parts = []
    for wp, name in parts:
        shape = wp.val().wrapped
        rotated = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
        new_wp = cq.Workplane("XY").newObject([cq.Shape(rotated)])
        rotated_parts.append((new_wp, name))

    return rotated_parts


# ===================================================================
# File loading
# ===================================================================

def load_cad_file(filepath):
    """Load a CAD file and return a CadQuery Workplane wrapper.

    For STEP files, uses the OCP reader directly for faster loading when
    multiple files are loaded in parallel (avoids CadQuery overhead).
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".step", ".stp"):
        from OCP.STEPControl import STEPControl_Reader
        reader = STEPControl_Reader()
        status = reader.ReadFile(filepath)
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEP read failed for {filepath}")
        reader.TransferRoots()
        shape = reader.OneShape()
        return cq.Workplane("XY").newObject([cq.Shape(shape)])
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


def _mesh_shell_to_solid(shape):
    """Convert a mesh/shell TopoDS_Shape to a solid via sewing.

    This is required so that boolean operations (cut) work on mesh-origin data.
    Returns the solid shape, or the original shape if conversion fails.

    Tries progressively larger sewing tolerances to handle meshes with gaps.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND
    from OCP.TopoDS import TopoDS

    # If already a solid, nothing to do
    if shape.ShapeType() == TopAbs_SOLID:
        return shape

    for sew_tol in (1e-6, 1e-4, 1e-2):
        try:
            # Sew the faces into a shell
            sew = BRepBuilderAPI_Sewing(sew_tol)
            sew.Add(shape)
            sew.Perform()
            sewn = sew.SewedShape()

            # Try to extract a shell and make a solid
            if sewn.ShapeType() == TopAbs_SHELL:
                maker = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(sewn))
                if maker.IsDone():
                    return maker.Solid()
            elif sewn.ShapeType() == TopAbs_SOLID:
                return sewn
            elif sewn.ShapeType() == TopAbs_COMPOUND:
                # Try all shells inside the compound
                explorer = TopExp_Explorer(sewn, TopAbs_SHELL)
                while explorer.More():
                    shell = TopoDS.Shell_s(explorer.Current())
                    maker = BRepBuilderAPI_MakeSolid(shell)
                    if maker.IsDone():
                        return maker.Solid()
                    explorer.Next()
        except Exception:
            continue

    return shape


def load_mesh_file(filepath):
    """Load a mesh file and convert to a CadQuery solid shape.

    Supports .stl, .obj, .ply, .3mf. Non-STL formats are converted to STL
    via trimesh first, then loaded through OCP and sewn into a solid for
    compatibility with boolean operations (cutting).
    """
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
        solid = _mesh_shell_to_solid(shape)
        return cq.Workplane("XY").newObject([cq.Shape(solid)])
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
    r"(?P<tier>inner|mid|outer)[_\- ]?(?P<levels>\d+(?:[_\- ]\d+)*)(?P<segment>[ab])?",
    re.IGNORECASE,
)


def parse_part_name(name):
    """Parse a part name into (tier, levels, segment).

    Supports multi-level spans: ``inner_2_3`` means inner spanning levels 2-3.
    Supports segment splitting: ``inner_2a`` / ``inner_2b`` splits the inner
    portion of level 2 into left (a) and right (b) halves.

    Returns:
        (tier, levels, segment): tier is "outer"/"mid"/"inner", levels is a
        list of ints, segment is "a", "b", or None.
        If the name doesn't match, returns (None, None, None).

    Examples:
        "outer_1"          -> ("outer", [1], None)
        "inner_2_3_steel"  -> ("inner", [2, 3], None)
        "mid_1_2_3"        -> ("mid", [1, 2, 3], None)
        "inner_2a"         -> ("inner", [2], "a")
        "inner_2b"         -> ("inner", [2], "b")
        "plate"            -> (None, None, None)
    """
    m = TIER_PATTERN.search(name)
    if m:
        tier = m.group("tier").lower()
        levels_str = m.group("levels")
        levels = [int(x) for x in re.split(r"[_\- ]", levels_str)]
        segment = m.group("segment")
        if segment:
            segment = segment.lower()
        return tier, levels, segment
    return None, None, None


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

    Segment splitting: parts named with an "a" or "b" suffix (e.g. inner_2a,
    inner_2b) are placed at the same position.  The segment tag is carried
    through so the cutting phase can split them into left/right halves.

    Returns a CadQuery Assembly and a list of
    (name, ocp_shape, location, rgb_tuple, segment) tuples.
    """
    # --- Classify parts by tier and levels ---
    classified = []  # (tier, levels_list, segment, wp, name, original_index)
    auto_level = 1

    for i, (wp, name) in enumerate(parts):
        tier, levels, segment = parse_part_name(name)
        if tier is None:
            # Unrecognized naming — treat as sequential outer parts
            tier = "outer"
            levels = [auto_level]
            segment = None
            auto_level += 1
        classified.append((tier, levels, segment, wp, name, i))

    # --- Build the set of levels and group by (level, tier) ---
    # Single-level parts go into levels_map[level][tier] as a list.
    # Multi-level (spanning) parts are collected separately.
    levels_map = {}  # level -> {tier -> [(wp, name, original_index, segment), ...]}
    spanning_parts = []  # (tier, levels_list, segment, wp, name, original_index)

    for tier, levels, segment, wp, name, idx in classified:
        if len(levels) == 1:
            levels_map.setdefault(levels[0], {}).setdefault(tier, []).append(
                (wp, name, idx, segment)
            )
        else:
            spanning_parts.append((tier, levels, segment, wp, name, idx))
            # Ensure all spanned levels exist in levels_map
            for lv in levels:
                levels_map.setdefault(lv, {})

    sorted_levels = sorted(levels_map.keys())

    # --- Stack outer parts vertically, center mid/inner within them ---
    assy = Assembly()
    part_info = []
    z_cursor = 0.0  # current Z position for stacking (always Z-axis for vertical)
    level_z_base = {}  # level -> z_cursor at bottom of that level
    level_z_top = {}   # level -> z at top of that level

    for level in sorted_levels:
        tier_group = levels_map[level]
        level_z_base[level] = z_cursor

        # Skip empty levels (only referenced by spanning parts, no own parts)
        if not tier_group:
            level_z_top[level] = z_cursor
            continue

        # Get the outer part for this level (required for positioning)
        # Use the first entry in the list for the reference part.
        ref_entry = None
        for t in ("outer", "mid", "inner"):
            if t in tier_group:
                ref_entry = tier_group[t][0]
                break
        if ref_entry is None:
            level_z_top[level] = z_cursor
            continue

        outer_wp = ref_entry[0]
        outer_shape = outer_wp.val().wrapped
        oxn, oyn, ozn, oxx, oyx, ozx = get_bounding_box(outer_shape)
        outer_height = ozx - ozn

        # Process each tier at this level
        for tier in ("outer", "mid", "inner"):
            if tier not in tier_group:
                continue
            for wp, name, idx, segment in tier_group[tier]:
                shape = wp.val().wrapped
                pxn, pyn, pzn, pxx, pyx, pzx = get_bounding_box(shape)
                part_cx = (pxn + pxx) / 2.0
                part_cy = (pyn + pyx) / 2.0

                dx = -part_cx  # center XY at origin
                dy = -part_cy
                dz = z_cursor - pzn

                loc = Location(Vector(dx, dy, dz))
                cq_color, rgb = pick_color(name, idx)

                assy.add(wp, name=name, loc=loc, color=cq_color)
                part_info.append((name, shape, loc, rgb, segment))

        # Advance the Z cursor past the outer part (+ gap)
        level_z_top[level] = z_cursor + outer_height
        z_cursor += outer_height + gap

    # --- Place spanning parts (parts that cover multiple levels) ---
    for tier, levels, segment, wp, name, idx in spanning_parts:
        first_level = min(levels)
        last_level = max(levels)

        if first_level not in level_z_base or last_level not in level_z_top:
            continue

        span_z_base = level_z_base[first_level]

        shape = wp.val().wrapped
        pxn, pyn, pzn, pxx, pyx, pzx = get_bounding_box(shape)
        part_cx = (pxn + pxx) / 2.0
        part_cy = (pyn + pyx) / 2.0

        # Center XY at origin, place Z-base at the start of the first level
        dx = -part_cx
        dy = -part_cy
        dz = span_z_base - pzn

        loc = Location(Vector(dx, dy, dz))
        cq_color, rgb = pick_color(name, idx)

        assy.add(wp, name=name, loc=loc, color=cq_color)
        part_info.append((name, shape, loc, rgb, segment))

    return assy, part_info


# ===================================================================
# Cutting
# ===================================================================

def _cutter_params(bbox_vals, axis_vec):
    """Compute radius, height, origin, and direction for a cutter from bbox."""
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
    return radius, height, origin, direction


def make_cutter(bbox_vals, angle, axis_vec):
    """
    Build a cylindrical-sector cutter from the global bounding box.
    The cutter spans `angle` degrees as a wedge, oriented along the stacking axis.
    """
    radius, height, origin, direction = _cutter_params(bbox_vals, axis_vec)
    ax = gp_Ax2(origin, direction)
    return BRepPrimAPI_MakeCylinder(ax, radius, height, math.radians(angle)).Shape()


def make_segment_cutter(bbox_vals, cut_angle, axis_vec, segment):
    """Build a cutter that keeps only the 'a' or 'b' half of the remaining arc.

    After removing ``cut_angle`` degrees (the main wedge), the remaining arc of
    ``360 - cut_angle`` degrees is split at its midpoint:

    * **segment "a"** (left half): keeps ``cut_angle .. midpoint``
    * **segment "b"** (right half): keeps ``midpoint .. 360``

    The function returns a compound of one or two cylindrical-sector cutters
    whose union removes everything *except* the desired half.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

    radius, height, origin, direction = _cutter_params(bbox_vals, axis_vec)
    remaining = 360.0 - cut_angle
    midpoint = cut_angle + remaining / 2.0  # angular midpoint of remaining arc

    if segment == "b":
        # Keep midpoint..360 → remove 0..midpoint
        ax = gp_Ax2(origin, direction)
        return BRepPrimAPI_MakeCylinder(
            ax, radius, height, math.radians(midpoint)
        ).Shape()
    else:
        # segment == "a"
        # Keep cut_angle..midpoint → remove 0..cut_angle AND midpoint..360
        # Cutter 1: the main wedge 0..cut_angle
        ax1 = gp_Ax2(origin, direction)
        c1 = BRepPrimAPI_MakeCylinder(
            ax1, radius, height, math.radians(cut_angle)
        ).Shape()
        # Cutter 2: the "b" region midpoint..360
        b_span = 360.0 - midpoint
        if b_span < 0.01:
            return c1
        # Rotate the coordinate system by midpoint degrees around the axis
        trsf = gp_Trsf()
        trsf.SetRotation(
            _axis_line(origin, direction), math.radians(midpoint)
        )
        ax2 = gp_Ax2(origin, direction)
        ax2.Transform(trsf)
        c2 = BRepPrimAPI_MakeCylinder(
            ax2, radius, height, math.radians(b_span)
        ).Shape()
        # Fuse the two cutters into one compound
        fuse = BRepAlgoAPI_Fuse(c1, c2)
        fuse.Build()
        if fuse.IsDone():
            return fuse.Shape()
        # Fallback: return them as a compound
        builder = BRep_Builder()
        comp = TopoDS_Compound()
        builder.MakeCompound(comp)
        builder.Add(comp, c1)
        builder.Add(comp, c2)
        return comp


def _axis_line(origin, direction):
    """Create a gp_Ax1 from origin point and direction for rotation."""
    from OCP.gp import gp_Ax1
    return gp_Ax1(origin, direction)


def cut_assembly(assy_compound, cutter_shape):
    """Boolean-cut the assembly compound with the cutter.

    Uses progressive fuzzy tolerance for robustness with mixed CAD/mesh geometry.
    Validates each result with BRepCheck_Analyzer and retries with a larger
    tolerance when the shape is geometrically invalid (e.g. non-manifold edges
    produced by near-coplanar faces at tight tolerances).
    """
    from OCP.TopTools import TopTools_ListOfShape
    from OCP.BRepCheck import BRepCheck_Analyzer

    args_list = TopTools_ListOfShape()
    args_list.Append(assy_compound)
    tools_list = TopTools_ListOfShape()
    tools_list.Append(cutter_shape)

    # Progressively looser tolerances; first attempt uses parallel mode for speed.
    tolerances = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    last_result = None

    for i, tol in enumerate(tolerances):
        op = BRepAlgoAPI_Cut()
        op.SetArguments(args_list)
        op.SetTools(tools_list)
        op.SetFuzzyValue(tol)
        op.SetRunParallel(i == 0)   # parallel only on first attempt
        op.SetNonDestructive(True)
        op.Build()

        if not op.IsDone():
            continue

        result = op.Shape()
        if result is None or result.IsNull():
            continue

        # Accept the result only if the topology is valid; otherwise retry with
        # a larger tolerance which smooths over near-coincident geometry.
        if BRepCheck_Analyzer(result).IsValid():
            return result

        # Keep the best (most recent) result in case all tolerances fail validation
        last_result = result

    # If no tolerance produced a valid shape, try to heal the best result we got
    if last_result is not None:
        try:
            from OCP.ShapeFix import ShapeFix_Shape
            fixer = ShapeFix_Shape(last_result)
            fixer.Perform()
            fixed = fixer.Shape()
            if fixed is not None and not fixed.IsNull():
                return fixed
        except Exception:
            pass
        return last_result

    raise RuntimeError("Boolean cut operation failed after all tolerance attempts")


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
        for i, entry in enumerate(parts_data):
            # Support both 4-tuple and 5-tuple (with segment) formats
            name, shape, loc, rgb = entry[0], entry[1], entry[2], entry[3]
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
    # 1. Expand globs and load all parts
    filepaths = expand_inputs(args.inputs)
    if not filepaths:
        print("No matching files found. Exiting.")
        return 1

    # Filter out missing files before loading
    valid_paths = []
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"  WARNING: '{filepath}' not found, skipping.")
        else:
            valid_paths.append(filepath)

    print(f"Loading {len(valid_paths)} part(s)...")
    parts = []
    if len(valid_paths) > 1:
        # Parallel loading for multiple files
        with ThreadPoolExecutor(max_workers=min(len(valid_paths), os.cpu_count() or 4)) as pool:
            future_map = {pool.submit(load_part, fp): fp for fp in valid_paths}
            # Collect results in original order
            results = {}
            for future in as_completed(future_map):
                fp = future_map[future]
                try:
                    wp, name = future.result()
                    results[fp] = (wp, name)
                    print(f"  Loaded: {name}")
                except Exception as e:
                    print(f"  ERROR loading '{fp}': {e}")
            # Preserve input order
            for fp in valid_paths:
                if fp in results:
                    parts.append(results[fp])
    else:
        for filepath in valid_paths:
            try:
                wp, name = load_part(filepath)
                parts.append((wp, name))
                print(f"  Loaded: {name}")
            except Exception as e:
                print(f"  ERROR loading '{filepath}': {e}")

    if not parts:
        print("No valid parts loaded. Exiting.")
        return 1

    # 1b. Cylinder orientation (optional)
    if getattr(args, "cyl", False):
        print("\nOrient parts for cylinder (--cyl): aligning cylinder axis with Z...")
        parts = orient_to_cylinder(parts)

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

        # Build compound and compute global bounding box
        print(f"\nCutting assembly at {args.cut_angle} degrees...")
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        moved_parts = []
        for name, shape, loc, color, segment in part_info:
            moved = apply_location(shape, loc)
            builder.Add(compound, moved)
            moved_parts.append((name, moved, segment))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()
        cutter = make_cutter(bbox_vals, args.cut_angle, axis_vec)

        # Pre-build segment cutters if any parts use a/b splitting
        has_segments = any(seg is not None for _, _, seg in moved_parts)
        seg_cutters = {}
        if has_segments:
            for seg in ("a", "b"):
                seg_cutters[seg] = make_segment_cutter(
                    bbox_vals, args.cut_angle, axis_vec, seg
                )

        # Cut each part individually for robustness with mixed geometry
        cut_builder = BRep_Builder()
        cut_compound = TopoDS_Compound()
        cut_builder.MakeCompound(cut_compound)
        cut_ok = 0
        cut_skip = 0

        for pname, moved_shape, segment in moved_parts:
            try:
                if segment in seg_cutters:
                    cut_part = cut_assembly(moved_shape, seg_cutters[segment])
                else:
                    cut_part = cut_assembly(moved_shape, cutter)
                # Verify the result is not empty
                part_bbox = Bnd_Box()
                BRepBndLib.Add_s(cut_part, part_bbox)
                if not part_bbox.IsVoid():
                    cut_builder.Add(cut_compound, cut_part)
                    cut_ok += 1
                else:
                    cut_skip += 1
            except Exception as e:
                print(f"  Warning: cut failed for '{pname}': {e}")
                # Include the original uncut part
                cut_builder.Add(cut_compound, moved_shape)
                cut_ok += 1

        cut_result_shape = cut_compound
        print(f"  Cut complete ({cut_ok} parts cut, {cut_skip} fully removed).")

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
            for name, shape, loc, color, _seg in part_info:
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
    print(f"  Cyl orient: {getattr(args, 'cyl', False)}")
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
        "--cyl", action="store_true",
        help="Orient parts so the cylinder height axis aligns with Z before stacking.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output",
    )
    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
