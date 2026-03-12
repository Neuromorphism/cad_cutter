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
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID

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

def orient_to_cylinder(parts, gap=0.0):
    """Rotate, orient, and restack parts so the cylinder height axis aligns with Z.

    Steps:
    1. PCA on combined vertices → find cylinder height axis → rotate to Z.
    2. For conic shapes, ensure the wider end is at the bottom (flip 180° if not).
    3. Sort parts by centroid position along the new Z axis.
    4. Restack sequentially: each part's bottom placed directly on top of the
       previous part, with *gap* spacing between them.

    Returns a list of (cq.Workplane, name) tuples ready for normal stacking.
    """
    if not parts:
        return parts

    # ------------------------------------------------------------------
    # Step 1 – PCA to find the cylinder height axis and rotate to Z
    # ------------------------------------------------------------------
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

    cov = (centered.T @ centered) / len(vertices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # Prefer the +Z half-space for a consistent initial orientation
    if principal[2] < 0:
        principal = -principal

    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(principal, z), -1.0, 1.0))

    px, py, pz = principal
    print(f"  Cylinder axis detected: ({px:.3f}, {py:.3f}, {pz:.3f})")

    current_parts = parts
    if abs(dot) < 1.0 - 1e-6:
        rot_axis = np.cross(principal, z)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        angle = math.acos(dot)
        print(f"  Rotating {math.degrees(angle):.1f}° to align cylinder axis with Z.")

        trsf = gp_Trsf()
        trsf.SetRotation(
            gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(*rot_axis.tolist())),
            angle,
        )
        current_parts = []
        for wp, name in parts:
            shape = BRepBuilderAPI_Transform(wp.val().wrapped, trsf, True).Shape()
            current_parts.append((cq.Workplane("XY").newObject([cq.Shape(shape)]), name))
    else:
        print("  Parts already aligned with Z-axis.")

    # ------------------------------------------------------------------
    # Step 2 – Wider end down: compare average radial distance from the
    #          Z-axis in the bottom and top halves of the combined vertex cloud.
    # ------------------------------------------------------------------
    verts_z = []
    for wp, name in current_parts:
        verts, _ = tessellate_shape(wp.val().wrapped, tolerance=1.0)
        if len(verts) > 0:
            verts_z.append(verts)

    if verts_z:
        vz = np.vstack(verts_z)
        z_vals = vz[:, 2]
        z_mid = (z_vals.min() + z_vals.max()) / 2.0
        bot_mask = z_vals < z_mid
        top_mask = z_vals >= z_mid

        if bot_mask.any() and top_mask.any():
            r_bot = np.mean(np.hypot(vz[bot_mask, 0], vz[bot_mask, 1]))
            r_top = np.mean(np.hypot(vz[top_mask, 0], vz[top_mask, 1]))

            if r_top > r_bot * (1.0 + 1e-3):
                print(
                    f"  Conic: wide end is up "
                    f"(r_bottom={r_bot:.3f}, r_top={r_top:.3f}), flipping 180°."
                )
                flip = gp_Trsf()
                flip.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.pi)
                flipped = []
                for wp, name in current_parts:
                    s = BRepBuilderAPI_Transform(wp.val().wrapped, flip, True).Shape()
                    flipped.append((cq.Workplane("XY").newObject([cq.Shape(s)]), name))
                current_parts = flipped

    # ------------------------------------------------------------------
    # Step 3 – Sort by centroid Z so parts are in natural stacking order
    # ------------------------------------------------------------------
    def _cz(wp_name):
        shape = wp_name[0].val().wrapped
        xn, yn, zn, xx, yx, zx = get_bounding_box(shape)
        return (zn + zx) / 2.0

    current_parts.sort(key=_cz)

    # ------------------------------------------------------------------
    # Step 4 – Restack: each part's bottom sits on top of the previous
    #          part, with *gap* between adjacent parts.
    # ------------------------------------------------------------------
    restacked = []
    z_cursor = 0.0
    for wp, name in current_parts:
        shape = wp.val().wrapped
        xn, yn, zn, xx, yx, zx = get_bounding_box(shape)
        height = zx - zn
        dz = z_cursor - zn
        if abs(dz) > 1e-9:
            t = gp_Trsf()
            t.SetTranslation(gp_Vec(0.0, 0.0, dz))
            shape = BRepBuilderAPI_Transform(shape, t, True).Shape()
        restacked.append((cq.Workplane("XY").newObject([cq.Shape(shape)]), name))
        z_cursor += height + gap

    return restacked


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
    """Compute radius, height, origin, and direction for a cutter from bbox.

    The cylinder axis passes through the XY centroid of the bounding box
    (in the two axes perpendicular to the stacking direction) so that the
    sector cleanly bisects the model even when it is not centred at the
    world origin.
    """
    xn, yn, zn, xx, yx, zx = bbox_vals

    if axis_vec.y:
        cx = (xn + xx) / 2.0
        cz = (zn + zx) / 2.0
        radius = max(abs(xn - cx), abs(xx - cx),
                     abs(zn - cz), abs(zx - cz)) * 2.5
        height = (yx - yn) * 3.0
        start = yn - (yx - yn)
        origin = gp_Pnt(cx, start, cz)
        direction = gp_Dir(0, 1, 0)
    elif axis_vec.z:
        cx = (xn + xx) / 2.0
        cy = (yn + yx) / 2.0
        radius = max(abs(xn - cx), abs(xx - cx),
                     abs(yn - cy), abs(yx - cy)) * 2.5
        height = (zx - zn) * 3.0
        start = zn - (zx - zn)
        origin = gp_Pnt(cx, cy, start)
        direction = gp_Dir(0, 0, 1)
    else:
        cy = (yn + yx) / 2.0
        cz = (zn + zx) / 2.0
        radius = max(abs(yn - cy), abs(yx - cy),
                     abs(zn - cz), abs(zx - cz)) * 2.5
        height = (xx - xn) * 3.0
        start = xn - (xx - xn)
        origin = gp_Pnt(start, cy, cz)
        direction = gp_Dir(1, 0, 0)

    radius = max(radius, 1.0)
    height = max(height, 1.0)
    return radius, height, origin, direction


def make_cutter(bbox_vals, angle, axis_vec):
    """
    Build a cylindrical-sector cutter from the global bounding box.
    The cutter spans ``angle`` degrees as a wedge, oriented along the
    stacking axis.

    For angles > 180° the cutter is built by fusing two sectors (each
    ≤ 180°) so that the boolean engine never has to subtract a single
    very-large sector — this avoids artifacts with complex geometry.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

    radius, height, origin, direction = _cutter_params(bbox_vals, axis_vec)

    if angle <= 180.0:
        ax = gp_Ax2(origin, direction)
        return BRepPrimAPI_MakeCylinder(
            ax, radius, height, math.radians(angle)
        ).Shape()

    # --- angle > 180°: build two sectors and fuse them ---
    # Sector 1: 0° → 180°
    ax1 = gp_Ax2(origin, direction)
    c1 = BRepPrimAPI_MakeCylinder(
        ax1, radius, height, math.radians(180)
    ).Shape()

    # Sector 2: 180° → angle  (span = angle − 180°)
    remaining = angle - 180.0
    trsf = gp_Trsf()
    trsf.SetRotation(
        _axis_line(origin, direction), math.radians(180)
    )
    ax2 = gp_Ax2(origin, direction)
    ax2.Transform(trsf)
    c2 = BRepPrimAPI_MakeCylinder(
        ax2, radius, height, math.radians(remaining)
    ).Shape()

    fuse = BRepAlgoAPI_Fuse(c1, c2)
    fuse.Build()
    if fuse.IsDone():
        return fuse.Shape()

    # Fallback: return as a compound
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    builder.Add(comp, c1)
    builder.Add(comp, c2)
    return comp


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


# -------------------------------------------------------------------
# Direct geometry cut  (split-and-filter, no boolean subtraction)
# -------------------------------------------------------------------

def _cut_half_planes(cut_angle, axis_vec, origin_pt):
    """Build two large planar faces at angle 0 and *cut_angle* around *axis_vec*.

    The planes pass through *origin_pt* and the stacking axis.  Together they
    define the boundaries of the removed wedge sector ``[0, cut_angle]``.

    Returns ``(face_at_0, face_at_cut_angle)``.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pln, gp_Ax3

    cut_rad = math.radians(cut_angle)

    # Perpendicular basis vectors in the plane normal to the stacking axis.
    if axis_vec.z:
        p1 = (1, 0, 0)
        p2 = (0, 1, 0)
    elif axis_vec.y:
        p1 = (1, 0, 0)
        p2 = (0, 0, 1)
    else:
        p1 = (0, 1, 0)
        p2 = (0, 0, 1)

    ox, oy, oz = origin_pt.X(), origin_pt.Y(), origin_pt.Z()

    # Plane at angle 0: contains the axis and perp1.  Normal = perp2.
    n1 = gp_Dir(p2[0], p2[1], p2[2])
    pln1 = gp_Pln(gp_Ax3(gp_Pnt(ox, oy, oz), n1))
    face1 = BRepBuilderAPI_MakeFace(pln1, -1e4, 1e4, -1e4, 1e4).Face()

    # Plane at cut_angle: normal = sin(cut)*perp1 − cos(cut)*perp2.
    n2x = math.sin(cut_rad) * p1[0] - math.cos(cut_rad) * p2[0]
    n2y = math.sin(cut_rad) * p1[1] - math.cos(cut_rad) * p2[1]
    n2z = math.sin(cut_rad) * p1[2] - math.cos(cut_rad) * p2[2]
    n2 = gp_Dir(n2x, n2y, n2z)
    pln2 = gp_Pln(gp_Ax3(gp_Pnt(ox, oy, oz), n2))
    face2 = BRepBuilderAPI_MakeFace(pln2, -1e4, 1e4, -1e4, 1e4).Face()

    return face1, face2


def _solid_in_sector(solid, start_angle, end_angle, axis_vec, origin_pt):
    """Return True if *solid* lies primarily inside the angular sector
    ``[start_angle, end_angle)`` (degrees) around *axis_vec*.

    Uses surface point sampling for robustness.  A coarse tessellation of
    the solid produces vertices distributed across its surface; each vertex
    is projected onto the radial plane and its angle tested.  The solid is
    classified by majority vote, making the result reliable even for
    thin-wall annular geometry where the centre-of-mass may sit near the
    axis or at a misleading angle.
    """
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    ox, oy, oz = origin_pt.X(), origin_pt.Y(), origin_pt.Z()
    eps = 0.1  # tolerance in degrees

    def _angle_in_sector(a):
        if start_angle < end_angle:
            return (start_angle + eps) < a < (end_angle - eps)
        return a > (start_angle + eps) or a < (end_angle - eps)

    # Tessellate and sample surface vertices
    try:
        verts, faces = tessellate_shape(solid, tolerance=0.5)
    except Exception:
        verts = np.zeros((0, 3))

    if len(verts) >= 4:
        # Project vertices onto radial plane and classify.
        # Only count vertices that are clearly inside or outside
        # the sector — skip those on or very near the boundary
        # planes (at start_angle or end_angle) as they belong to
        # both adjacent sectors after splitting.
        if axis_vec.z:
            px = verts[:, 0] - ox
            py = verts[:, 1] - oy
        elif axis_vec.y:
            px = verts[:, 0] - ox
            py = verts[:, 2] - oz
        else:
            px = verts[:, 1] - oy
            py = verts[:, 2] - oz

        angles = np.degrees(np.arctan2(py, px)) % 360.0
        boundary_tol = 2.0  # degrees — skip vertices near the boundaries
        votes_in = 0
        votes_out = 0
        for a in angles:
            a = float(a)
            # Skip vertices on or very near the sector boundaries
            near_start = min(abs(a - start_angle), abs(a - start_angle - 360),
                             abs(a - start_angle + 360)) < boundary_tol
            near_end = min(abs(a - end_angle), abs(a - end_angle - 360),
                           abs(a - end_angle + 360)) < boundary_tol
            if near_start or near_end:
                continue
            if _angle_in_sector(a):
                votes_in += 1
            else:
                votes_out += 1

        if votes_in + votes_out > 0:
            return votes_in > votes_out

    # Fallback: use centre-of-mass
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)
    com = props.CentreOfMass()

    if axis_vec.z:
        cx, cy = com.X() - ox, com.Y() - oy
    elif axis_vec.y:
        cx, cy = com.X() - ox, com.Z() - oz
    else:
        cx, cy = com.Y() - oy, com.Z() - oz

    angle = math.degrees(math.atan2(cy, cx)) % 360.0
    return _angle_in_sector(angle)


def cut_part_direct(shape, cut_angle, axis_vec, origin_pt):
    """Cut *shape* by splitting it with half-planes and discarding the wedge.

    Instead of a boolean subtraction (``BRepAlgoAPI_Cut``) this:

    1. Creates two planar faces at the cut boundaries (angle 0 and
       *cut_angle*).
    2. Splits the shape along these planes with ``BRepAlgoAPI_Splitter``,
       which re-trims the underlying surface functions and inserts new
       planar faces at the cut boundaries automatically.
    3. Filters the resulting solids, keeping only those whose centre of
       mass lies outside the removed sector ``(0, cut_angle)``.

    For *cut_angle* near 180° the two cutting planes are geometrically
    identical (parallel normals).  In this case only one plane is used and
    solids are filtered by which side of the Y-axis (in the radial plane)
    they fall on.

    Returns the kept compound (may contain one or more solids).
    Raises ``RuntimeError`` if the splitter operation fails.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
    from OCP.TopTools import TopTools_ListOfShape

    face1, face2 = _cut_half_planes(cut_angle, axis_vec, origin_pt)

    args = TopTools_ListOfShape()
    args.Append(shape)
    tools = TopTools_ListOfShape()
    tools.Append(face1)

    # At exactly 180° the two cutting planes are geometrically identical
    # (anti-parallel normals).  Use only one plane and filter by half-space.
    near_180 = abs(cut_angle - 180.0) < 0.5
    if not near_180:
        tools.Append(face2)

    splitter = BRepAlgoAPI_Splitter()
    splitter.SetArguments(args)
    splitter.SetTools(tools)
    splitter.Build()

    if not splitter.IsDone():
        raise RuntimeError("BRepAlgoAPI_Splitter failed on shape")

    result = splitter.Shape()

    # Collect solids and keep those outside the removed sector.
    exp = TopExp_Explorer(result, TopAbs_SOLID)
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    kept = 0

    if near_180:
        # 180° special case: one plane divides the shape in half.
        # The removed sector [0°, 180°] is the positive-perp2 side.
        # Keep solids whose centre-of-mass is on the negative-perp2 side.
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp

        ox, oy, oz = origin_pt.X(), origin_pt.Y(), origin_pt.Z()

        while exp.More():
            solid = TopoDS.Solid_s(exp.Current())
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(solid, props)
            com = props.CentreOfMass()

            # The plane at 0° has normal = perp2.  Solids in the removed
            # sector [0°, 180°] have positive perp2 coordinate.
            if axis_vec.z:
                coord = com.Y() - oy  # perp2 = Y
            elif axis_vec.y:
                coord = com.Z() - oz  # perp2 = Z
            else:
                coord = com.Z() - oz  # perp2 = Z

            if coord < 0:
                # Negative side — keep (outside the removed sector)
                builder.Add(compound, solid)
                kept += 1
            exp.Next()
    else:
        while exp.More():
            solid = TopoDS.Solid_s(exp.Current())
            if not _solid_in_sector(solid, 0.0, cut_angle, axis_vec, origin_pt):
                builder.Add(compound, solid)
                kept += 1
            exp.Next()

    if kept == 0:
        return None

    return compound


def cut_part_direct_segment(shape, cut_angle, axis_vec, origin_pt, segment):
    """Direct-geometry variant of segment splitting (a/b halves).

    After removing the main wedge ``[0, cut_angle]``, the remaining arc
    ``[cut_angle, 360]`` is bisected:

    * **segment "a"** keeps ``[cut_angle, midpoint]``
    * **segment "b"** keeps ``[midpoint, 360]``

    Three half-planes are used: at 0°, *cut_angle*, and *midpoint*.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
    from OCP.TopTools import TopTools_ListOfShape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pln, gp_Ax3

    remaining = 360.0 - cut_angle
    midpoint = cut_angle + remaining / 2.0

    # Build three half-planes: at 0, cut_angle, and midpoint.
    face_0, face_cut = _cut_half_planes(cut_angle, axis_vec, origin_pt)

    # Additional plane at midpoint
    mid_rad = math.radians(midpoint)
    if axis_vec.z:
        p1, p2 = (1, 0, 0), (0, 1, 0)
    elif axis_vec.y:
        p1, p2 = (1, 0, 0), (0, 0, 1)
    else:
        p1, p2 = (0, 1, 0), (0, 0, 1)

    ox, oy, oz = origin_pt.X(), origin_pt.Y(), origin_pt.Z()
    nmx = math.sin(mid_rad) * p1[0] - math.cos(mid_rad) * p2[0]
    nmy = math.sin(mid_rad) * p1[1] - math.cos(mid_rad) * p2[1]
    nmz = math.sin(mid_rad) * p1[2] - math.cos(mid_rad) * p2[2]
    nm = gp_Dir(nmx, nmy, nmz)
    pln_mid = gp_Pln(gp_Ax3(gp_Pnt(ox, oy, oz), nm))
    face_mid = BRepBuilderAPI_MakeFace(pln_mid, -1e4, 1e4, -1e4, 1e4).Face()

    # Split by all three planes
    args = TopTools_ListOfShape()
    args.Append(shape)
    tools = TopTools_ListOfShape()
    tools.Append(face_0)
    tools.Append(face_cut)
    tools.Append(face_mid)

    splitter = BRepAlgoAPI_Splitter()
    splitter.SetArguments(args)
    splitter.SetTools(tools)
    splitter.Build()

    if not splitter.IsDone():
        raise RuntimeError("BRepAlgoAPI_Splitter failed (segment split)")

    result = splitter.Shape()

    # Filter solids: segment "a" keeps [cut_angle, midpoint],
    #                segment "b" keeps [midpoint, 360].
    if segment == "a":
        keep_start, keep_end = cut_angle, midpoint
    else:
        keep_start, keep_end = midpoint, 360.0

    exp = TopExp_Explorer(result, TopAbs_SOLID)
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    kept = 0

    while exp.More():
        solid = TopoDS.Solid_s(exp.Current())
        if _solid_in_sector(solid, keep_start, keep_end, axis_vec, origin_pt):
            builder.Add(compound, solid)
            kept += 1
        exp.Next()

    if kept == 0:
        return None

    return compound


def cut_assembly(assy_compound, cutter_shape):
    """Boolean-cut the assembly compound with the cutter.

    Uses progressive fuzzy tolerance for robustness with mixed CAD/mesh
    geometry.  Each candidate result is validated in two ways:

    1. **Topological** — ``BRepCheck_Analyzer`` confirms the shape is
       well-formed (no degenerate edges, self-intersections, etc.).
    2. **Volumetric cross-check** — a result is only accepted when a
       *second* tolerance level produces a volume within 10 % of the
       first.  This catches the case where the boolean engine returns a
       topologically valid but geometrically *wrong* shape at a tight
       tolerance (e.g. leaving the model nearly intact).

    The first confirmed result (tightest tolerance with cross-check
    agreement) is returned.
    """
    from OCP.TopTools import TopTools_ListOfShape
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    args_list = TopTools_ListOfShape()
    args_list.Append(assy_compound)
    tools_list = TopTools_ListOfShape()
    tools_list.Append(cutter_shape)

    # Progressively looser tolerances; first attempt uses parallel mode for
    # speed.
    tolerances = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # Collect valid (topology-ok) candidates: (tolerance, shape, volume)
    candidates = []
    # Also track topology-failed candidates that might still be usable
    # after ShapeFix (common with thin-wall geometry).
    topo_failed = []
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

        # Compute volume regardless of topology validity
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(result, props)
        vol = props.Mass()

        if not BRepCheck_Analyzer(result).IsValid():
            last_result = result
            topo_failed.append((tol, result, vol))
            continue

        candidates.append((tol, result, vol))

        # Cross-check: two adjacent valid results must agree within 10 %
        if len(candidates) >= 2:
            prev_vol = candidates[-2][2]
            denom = max(abs(prev_vol), abs(vol), 1e-12)
            if abs(vol - prev_vol) / denom < 0.10:
                # Volumes agree — return the tighter-tolerance result
                return candidates[-2][1]
            else:
                # Disagreement — the earlier candidate was likely wrong;
                # discard it and keep going with the current one.
                candidates = [candidates[-1]]

    # If only one candidate was produced and it was never cross-checked,
    # accept it (better than nothing).
    if candidates:
        return candidates[-1][1]

    # No topologically valid results — try to heal the best topology-failed
    # candidate using ShapeFix.  Prefer candidates whose volumes agree
    # (cross-check among topo_failed entries).
    if topo_failed:
        from OCP.ShapeFix import ShapeFix_Shape

        # Try cross-checking topo-failed volumes
        for j in range(len(topo_failed) - 1):
            vol_a = topo_failed[j][2]
            vol_b = topo_failed[j + 1][2]
            denom = max(abs(vol_a), abs(vol_b), 1e-12)
            if abs(vol_a - vol_b) / denom < 0.10:
                # Volumes agree — fix the tighter-tolerance one
                best = topo_failed[j][1]
                try:
                    fixer = ShapeFix_Shape(best)
                    fixer.Perform()
                    fixed = fixer.Shape()
                    if fixed is not None and not fixed.IsNull():
                        return fixed
                except Exception:
                    pass
                return best

        # Single topo-failed result — try to fix it
        best = topo_failed[-1][1]
        try:
            fixer = ShapeFix_Shape(best)
            fixer.Perform()
            fixed = fixer.Shape()
            if fixed is not None and not fixed.IsNull():
                return fixed
        except Exception:
            pass
        return best

    if last_result is not None:
        return last_result

    raise RuntimeError("Boolean cut operation failed after all tolerance attempts")


# ===================================================================
# Physics simulation
# ===================================================================

def _shape_to_clean_trimesh(shape, tolerance=0.05, angular=0.5):
    """Convert an OCP shape to a cleaned trimesh, removing degenerate faces."""
    import trimesh

    verts, faces_pv = tessellate_shape(shape, tolerance=tolerance, angular=angular)
    if len(verts) == 0:
        return None

    # PyVista face format [3, i, j, k, ...] -> trimesh (N,3)
    tri_faces = []
    i = 0
    while i < len(faces_pv):
        n = faces_pv[i][0] if hasattr(faces_pv[i], '__len__') else faces_pv[i, 0]
        for t in range(1, n - 1):
            tri_faces.append([faces_pv[i, 1],
                              faces_pv[i, 1 + t],
                              faces_pv[i, 2 + t]])
        i += 1

    if not tri_faces:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(tri_faces),
                           process=False)

    # Remove degenerate faces (zero-area or collapsed)
    face_areas = mesh.area_faces
    valid_mask = face_areas > 0
    if not valid_mask.all():
        mesh.update_faces(valid_mask)
    mesh.remove_unreferenced_vertices()

    # Merge duplicate vertices and remove duplicate faces
    mesh.merge_vertices()

    # Repair normals and fill small holes
    mesh.fix_normals()

    if len(mesh.faces) == 0:
        return None

    # Make sure it's watertight for volume checks
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)

    return mesh


def _mesh_convex_hull(mesh):
    """Return the convex hull of a trimesh for fast collision checks."""
    try:
        return mesh.convex_hull
    except Exception:
        return mesh


def simulate_physics(part_info, axis_vec, gap, max_iters=50,
                     settle_threshold=1e-4, debug=False):
    """Run a quick gravity-settle simulation on assembled parts.

    Each part is dropped along the assembly axis until it rests on the floor
    (at the origin) or on another already-settled part.  Concave nesting is
    resolved by ray-casting through the mesh interiors, so hollow parts like
    outer_1/outer_2 will nest together and produce negative bounding-box gaps.

    Parts are tessellated into cleaned meshes (degenerate faces removed)
    before simulation.

    Returns an updated part_info list with adjusted locations reflecting the
    settled positions.
    """
    import trimesh

    if not part_info:
        return part_info

    # Determine axis index (0=x, 1=y, 2=z)
    if axis_vec.x:
        ax = 0
    elif axis_vec.y:
        ax = 1
    else:
        ax = 2

    # -- Build cleaned meshes and track offsets --
    bodies = []  # (name, mesh, offset[3], ocp_shape, loc, rgb, segment)
    for name, shape, loc, rgb, segment in part_info:
        moved = apply_location(shape, loc)
        mesh = _shape_to_clean_trimesh(moved)
        if mesh is None:
            bodies.append((name, None, np.zeros(3), shape, loc, rgb, segment))
            continue
        hull = _mesh_convex_hull(mesh)
        bodies.append((name, mesh, np.zeros(3), shape, loc, rgb, segment))

    if debug:
        print(f"  [PHYS] Settling {len(bodies)} bodies along axis={ax}")

    # -- Sort by position along axis (highest first = drop order) --
    def _axis_min(b):
        if b[1] is None:
            return 0.0
        return b[1].bounds[0, ax]

    bodies.sort(key=_axis_min)

    # Floor at z=0
    floor_z = 0.0

    # -- Drop each body iteratively --
    # Process from lowest to highest: the lowest body settles on the floor,
    # then each successive body settles on whatever is below it.
    settled_meshes = []  # list of meshes already in their final positions

    for idx in range(len(bodies)):
        name, mesh, offset, shape, loc, rgb, segment = bodies[idx]
        if mesh is None:
            continue

        current_mesh = mesh.copy()

        # Find the resting position by dropping along the axis
        # Start: current mesh position. Target: as low as possible.
        current_min = current_mesh.bounds[0, ax]

        # 1) Floor constraint: how far can we drop before hitting the floor?
        drop_to_floor = current_min - floor_z

        # 2) Check against all settled meshes for support surfaces
        best_drop = drop_to_floor  # maximum we can drop

        for settled in settled_meshes:
            # Quick AABB overlap check in the non-axis directions
            my_bounds = current_mesh.bounds
            ot_bounds = settled.bounds

            # Check overlap in the two non-axis dimensions
            other_axes = [d for d in range(3) if d != ax]
            overlap = True
            for oa in other_axes:
                if my_bounds[0, oa] >= ot_bounds[1, oa] or \
                   my_bounds[1, oa] <= ot_bounds[0, oa]:
                    overlap = False
                    break
            if not overlap:
                continue

            # Ray-cast to find support surface on the settled mesh
            support_drop = _find_support_drop(current_mesh, settled, ax)
            if support_drop is not None and support_drop < best_drop:
                best_drop = support_drop

        # Apply the drop
        if best_drop > settle_threshold:
            translation = np.zeros(3)
            translation[ax] = -best_drop
            current_mesh.apply_translation(translation)
            offset_new = offset.copy()
            offset_new[ax] -= best_drop
            bodies[idx] = (name, mesh, offset_new, shape, loc, rgb, segment)

            if debug:
                print(f"  [PHYS] '{name}': dropped {best_drop:.4f} along axis")
        else:
            if debug:
                print(f"  [PHYS] '{name}': already settled")

        # Add the settled mesh to the list (in final position)
        settled_copy = mesh.copy()
        final_offset = bodies[idx][2]
        settled_copy.apply_translation(final_offset)
        settled_meshes.append(settled_copy)

    # -- Build updated part_info with new locations --
    updated = []
    for body in bodies:
        name, mesh, offset, shape, loc, rgb, segment = body
        if mesh is None:
            updated.append((name, shape, loc, rgb, segment))
        else:
            dx, dy, dz = offset[0], offset[1], offset[2]
            if abs(dx) < 1e-8 and abs(dy) < 1e-8 and abs(dz) < 1e-8:
                updated.append((name, shape, loc, rgb, segment))
            else:
                # Compose: apply original loc, then translate by sim offset
                new_loc = Location(Vector(dx, dy, dz)) * loc
                updated.append((name, shape, new_loc, rgb, segment))

    # -- Report bbox gap changes --
    _report_nesting(updated, ax, gap, debug)

    return updated


def _find_support_drop(falling_mesh, settled_mesh, ax):
    """Find how far *falling_mesh* can drop along *ax* before resting on
    *settled_mesh*.

    Casts a grid of rays downward from the bottom face of the falling mesh
    through the settled mesh.  For each ray that hits the settled surface,
    the drop distance is the gap between the falling mesh bottom and the
    topmost hit (the support surface).

    For concave/hollow parts the rays may enter the interior and find an
    inner floor, enabling nesting.

    Returns the maximum safe drop distance, or None if there is no
    interaction (the meshes don't overlap in the lateral dimensions).
    """
    bounds_f = falling_mesh.bounds
    bounds_s = settled_mesh.bounds

    # Sample grid on the XY footprint of the falling mesh
    n_samples = 10
    other_axes = [d for d in range(3) if d != ax]
    margin_u = (bounds_f[1, other_axes[0]] - bounds_f[0, other_axes[0]]) * 0.02
    margin_v = (bounds_f[1, other_axes[1]] - bounds_f[0, other_axes[1]]) * 0.02

    u_range = np.linspace(bounds_f[0, other_axes[0]] + margin_u,
                          bounds_f[1, other_axes[0]] - margin_u, n_samples)
    v_range = np.linspace(bounds_f[0, other_axes[1]] + margin_v,
                          bounds_f[1, other_axes[1]] - margin_v, n_samples)

    origins = []
    dir_down = np.zeros(3)
    dir_down[ax] = -1.0

    for u in u_range:
        for v in v_range:
            o = np.zeros(3)
            o[other_axes[0]] = u
            o[other_axes[1]] = v
            o[ax] = bounds_f[0, ax]  # bottom of falling mesh
            origins.append(o)

    if not origins:
        return None

    origins = np.array(origins)
    directions = np.tile(dir_down, (len(origins), 1))

    # Cast rays downward against the settled mesh
    try:
        hits, ray_ids, _ = settled_mesh.ray.intersects_location(
            origins, directions, multiple_hits=True
        )
    except Exception:
        return None

    if len(hits) == 0:
        return None

    # For each ray, find the topmost hit on the settled mesh.
    # The hit with the highest coordinate along ax is the support surface
    # (closest point below the falling part).
    ray_support = {}
    for hit, rid in zip(hits, ray_ids):
        hit_val = hit[ax]
        if rid not in ray_support or hit_val > ray_support[rid]:
            ray_support[rid] = hit_val

    if not ray_support:
        return None

    # The support level is the highest surface point found.
    # The drop is how far the falling mesh bottom is above that surface.
    support_level = max(ray_support.values())
    drop = bounds_f[0, ax] - support_level

    if drop > 0:
        return drop

    return 0.0


def _report_nesting(part_info, ax, original_gap, debug):
    """Print nesting report showing effective gaps vs bounding-box gaps."""
    if not part_info:
        return

    # Compute bounding boxes of final positioned parts
    positioned = []
    for name, shape, loc, rgb, segment in part_info:
        moved = apply_location(shape, loc)
        bb = get_bounding_box(moved)
        # bb = (xmin, ymin, zmin, xmax, ymax, zmax)
        ax_min = bb[ax]
        ax_max = bb[ax + 3]
        positioned.append((name, ax_min, ax_max))

    positioned.sort(key=lambda x: x[1])

    print("\n  Physics nesting report:")
    print(f"  {'Part':<25s} {'AxisMin':>10s} {'AxisMax':>10s} {'Gap':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    prev_max = None
    for name, ax_min, ax_max in positioned:
        if prev_max is not None:
            gap_val = ax_min - prev_max
            gap_str = f"{gap_val:+.4f}"
        else:
            gap_str = "—"
        print(f"  {name:<25s} {ax_min:10.4f} {ax_max:10.4f} {gap_str:>10s}")
        if prev_max is None or ax_max > prev_max:
            prev_max = ax_max


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
        print("\nOrient parts for cylinder (--cyl)...")
        parts = orient_to_cylinder(parts, gap=args.gap)

    # 2. Stack parts
    axis_vec = AXIS_MAP[args.axis]
    print(f"\nStacking {len(parts)} part(s) along {args.axis.upper()}-axis (gap={args.gap})...")
    assy, part_info = stack_parts(parts, axis_vec, args.gap)
    print(f"  Assembly created with {len(part_info)} part(s).")

    # 2b. Physics simulation (optional)
    if getattr(args, "phys", False):
        print("\nRunning physics simulation (--phys)...")
        part_info = simulate_physics(
            part_info, axis_vec, args.gap,
            debug=getattr(args, "debug", False),
        )
        # Rebuild assembly with updated locations
        assy = Assembly()
        for name, shape, loc, rgb, segment in part_info:
            wp = cq.Workplane("XY").newObject([cq.Shape(shape)])
            cq_color = Color(*rgb) if len(rgb) == 3 else Color(*rgb[:3])
            assy.add(wp, name=name, loc=loc, color=cq_color)
        print("  Physics simulation complete.")

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
        use_direct = getattr(args, "cut_direct", False)
        method_label = "direct geometry" if use_direct else "boolean"
        print(f"\nCutting assembly at {args.cut_angle} degrees ({method_label})...")
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

        if args.debug:
            dbg_radius, dbg_height, dbg_origin, dbg_direction = _cutter_params(bbox_vals, axis_vec)
            print(f"  [DEBUG] Cutter params: radius={dbg_radius:.3f}, height={dbg_height:.3f}, "
                  f"origin=({dbg_origin.X():.3f}, {dbg_origin.Y():.3f}, {dbg_origin.Z():.3f}), "
                  f"direction=({dbg_direction.X():.3f}, {dbg_direction.Y():.3f}, {dbg_direction.Z():.3f}), "
                  f"angle={args.cut_angle} deg")
            print(f"  [DEBUG] bbox: xmin={bbox_vals[0]:.3f} ymin={bbox_vals[1]:.3f} zmin={bbox_vals[2]:.3f} "
                  f"xmax={bbox_vals[3]:.3f} ymax={bbox_vals[4]:.3f} zmax={bbox_vals[5]:.3f}")

        # Compute the origin point for direct cutting (XY centroid of bbox)
        _r, _h, origin_pt, _d = _cutter_params(bbox_vals, axis_vec)

        # Pre-build segment cutters / check for segments
        has_segments = any(seg is not None for _, _, seg in moved_parts)

        if not use_direct:
            cutter = make_cutter(bbox_vals, args.cut_angle, axis_vec)
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
                if use_direct:
                    # --- Direct geometry approach ---
                    if segment is not None:
                        cut_part = cut_part_direct_segment(
                            moved_shape, args.cut_angle, axis_vec,
                            origin_pt, segment,
                        )
                    else:
                        cut_part = cut_part_direct(
                            moved_shape, args.cut_angle, axis_vec,
                            origin_pt,
                        )
                else:
                    # --- Boolean approach ---
                    if segment in seg_cutters:
                        cut_part = cut_assembly(moved_shape, seg_cutters[segment])
                    else:
                        cut_part = cut_assembly(moved_shape, cutter)

                if cut_part is None:
                    cut_skip += 1
                    if args.debug:
                        print(f"  [DEBUG] '{pname}': fully removed by cutter.")
                    continue

                # Verify the result is not empty
                part_bbox = Bnd_Box()
                BRepBndLib.Add_s(cut_part, part_bbox)
                if not part_bbox.IsVoid():
                    cut_builder.Add(cut_compound, cut_part)
                    cut_ok += 1
                    if args.debug:
                        print(f"  [DEBUG] '{pname}': cut succeeded.")
                else:
                    cut_skip += 1
                    if args.debug:
                        print(f"  [DEBUG] '{pname}': fully removed by cutter.")
            except Exception as e:
                print(f"  Warning: cut failed for '{pname}': {e}")
                # Include the original uncut part
                cut_builder.Add(cut_compound, moved_shape)
                cut_ok += 1

        if args.debug and not use_direct:
            cutter = make_cutter(bbox_vals, args.cut_angle, axis_vec)
            print("  [DEBUG] Including cutter shape in output STEP as 'CUTTER_DEBUG'.")
            cut_builder.Add(cut_compound, cutter)

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
    print(f"  Physics:    {getattr(args, 'phys', False)}")
    print(f"  Axis:       {args.axis.upper()}")
    print(f"  Gap:        {args.gap}")
    if args.cut_angle is not None:
        cut_method = "direct" if getattr(args, "cut_direct", False) else "boolean"
        print(f"  Cut angle:  {args.cut_angle} deg ({cut_method})")
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
        "--cut-direct", action="store_true",
        help="Use direct geometry calculation for cutting instead of boolean "
             "subtraction.  Splits faces along cut planes and filters solids, "
             "which is faster and produces cleaner cap surfaces.",
    )
    parser.add_argument(
        "--phys", action="store_true",
        help="Run a quick physics simulation: stack parts and release under "
             "gravity so concentric parts nest together.  Reports effective "
             "gaps (which become negative when parts nest inside each other).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output",
    )
    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
