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
from OCP.BRepTools import BRepTools
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Ax2, gp_Pnt, gp_Dir
from OCP.gp import gp_GTrsf
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_GTransform
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
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


def get_tight_bounding_box(shape, tol=1.0):
    """Return (xmin,ymin,zmin, xmax,ymax,zmax) using the mesh triangulation.

    Some STEP files contain degenerate parametric faces (e.g. a BSpline with
    a tiny physical area but a huge parametric domain) whose exact bounding
    box inflates the overall bbox far beyond the visible geometry.
    Meshing first and querying with useTriangulation=True restricts the bbox
    to actual triangulated surfaces, which avoids this inflation.

    Falls back to the exact bbox if meshing produces a void result.
    """
    BRepMesh_IncrementalMesh(shape, tol)
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox, True)  # True = useTriangulation
    if bbox.IsVoid():
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


def scale_shape(shape, factor):
    """Return a uniformly scaled copy of the OCP shape.

    Scales about the origin by *factor*.
    """
    trsf = gp_Trsf()
    trsf.SetScaleFactor(factor)
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
    for entry in parts:
        wp, name = entry[0], entry[1]
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
        for entry in parts:
            wp, name = entry[0], entry[1]
            extra = entry[2:] if len(entry) > 2 else ()
            shape = BRepBuilderAPI_Transform(wp.val().wrapped, trsf, True).Shape()
            current_parts.append((cq.Workplane("XY").newObject([cq.Shape(shape)]), name) + extra)
    else:
        print("  Parts already aligned with Z-axis.")

    # ------------------------------------------------------------------
    # Step 2 – Wider end down: compare average radial distance from the
    #          Z-axis in the bottom and top halves of the combined vertex cloud.
    # ------------------------------------------------------------------
    verts_z = []
    for entry in current_parts:
        wp = entry[0]
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
                for entry in current_parts:
                    wp, name = entry[0], entry[1]
                    extra = entry[2:] if len(entry) > 2 else ()
                    s = BRepBuilderAPI_Transform(wp.val().wrapped, flip, True).Shape()
                    flipped.append((cq.Workplane("XY").newObject([cq.Shape(s)]), name) + extra)
                current_parts = flipped

    # ------------------------------------------------------------------
    # Step 3 – Sort by centroid Z so parts are in natural stacking order
    # ------------------------------------------------------------------
    def _cz(wp_name):
        shape = wp_name[0].val().wrapped
        xn, yn, zn, xx, yx, zx = get_tight_bounding_box(shape)
        return (zn + zx) / 2.0

    current_parts.sort(key=_cz)

    # ------------------------------------------------------------------
    # Step 4 – Restack: each part's bottom sits on top of the previous
    #          part, with *gap* between adjacent parts.
    # ------------------------------------------------------------------
    restacked = []
    z_cursor = 0.0
    for entry in current_parts:
        wp, name = entry[0], entry[1]
        extra = entry[2:] if len(entry) > 2 else ()
        shape = wp.val().wrapped
        xn, yn, zn, xx, yx, zx = get_tight_bounding_box(shape)
        height = zx - zn
        dz = z_cursor - zn
        if abs(dz) > 1e-9:
            t = gp_Trsf()
            t.SetTranslation(gp_Vec(0.0, 0.0, dz))
            shape = BRepBuilderAPI_Transform(shape, t, True).Shape()
        restacked.append((cq.Workplane("XY").newObject([cq.Shape(shape)]), name) + extra)
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
        # Fast path: read BREP natively through OCP to avoid CadQuery importer
        # overhead when loading many parts in parallel.
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        ok = BRepTools.Read_s(shape, filepath, builder)
        if not ok:
            raise RuntimeError(f"BREP read failed for {filepath}")
        return cq.Workplane("XY").newObject([cq.Shape(shape)])
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
                # Try all shells inside the compound and pick the largest
                from OCP.BRepBndLib import BRepBndLib as _BRepBndLib
                from OCP.Bnd import Bnd_Box as _Bnd_Box
                explorer = TopExp_Explorer(sewn, TopAbs_SHELL)
                best_solid = None
                best_diag = -1.0
                while explorer.More():
                    shell = TopoDS.Shell_s(explorer.Current())
                    maker = BRepBuilderAPI_MakeSolid(shell)
                    if maker.IsDone():
                        candidate = maker.Solid()
                        bb = _Bnd_Box()
                        _BRepBndLib.Add_s(candidate, bb)
                        if not bb.IsVoid():
                            vals = bb.Get()
                            diag = ((vals[3]-vals[0])**2
                                    + (vals[4]-vals[1])**2
                                    + (vals[5]-vals[2])**2) ** 0.5
                            if diag > best_diag:
                                best_diag = diag
                                best_solid = candidate
                    explorer.Next()
                if best_solid is not None:
                    return best_solid
        except Exception:
            continue

    return shape


def load_mesh_file(filepath, require_solid=True):
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
        out_shape = _mesh_shell_to_solid(shape) if require_solid else shape
        return cq.Workplane("XY").newObject([cq.Shape(out_shape)])
    finally:
        if tmp_stl:
            os.unlink(tmp_stl.name)


def load_part(filepath, require_solid=True):
    """Load any supported CAD or mesh file, return (cq.Workplane, name)."""
    ext = os.path.splitext(filepath)[1].lower()
    name = os.path.splitext(os.path.basename(filepath))[0]

    if ext in CAD_EXTENSIONS:
        wp = load_cad_file(filepath)
    elif ext in MESH_EXTENSIONS:
        wp = load_mesh_file(filepath, require_solid=require_solid)
    else:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: "
            f"{sorted(CAD_EXTENSIONS | MESH_EXTENSIONS)}"
        )
    return wp, name


def is_mesh_file(filepath):
    """Return True if *filepath* has a mesh extension (.stl, .obj, .ply, .3mf)."""
    return os.path.splitext(filepath)[1].lower() in MESH_EXTENSIONS


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
# Autoscale helpers
# ===================================================================

_DIAMETER_PATTERN = re.compile(r"_d(\d+(?:\.\d+)?)", re.IGNORECASE)


def _parse_target_diameter(name):
    """Extract a target diameter from a ``_dN`` suffix in *name*.

    Returns the numeric value as a float, or *None* if no such tag exists.

    Examples:
        "inner_2_3_d8"     -> 8.0
        "inner_2_3_d12.5"  -> 12.5
        "outer_1"          -> None
    """
    m = _DIAMETER_PATTERN.search(name)
    if m:
        return float(m.group(1))
    return None


def _get_xy_diameter(shape):
    """Return the XY-plane diameter of *shape* (max of width and depth)."""
    xn, yn, _, xx, yx, _ = get_tight_bounding_box(shape)
    return max(xx - xn, yx - yn)


# Thresholds used by autoscale unit detection.
# If inner diameter < outer diameter / INNER_TINY_RATIO, the inner is
# probably in mm while the outer is in inches.
_INNER_TINY_RATIO = 5.0
_MM_TO_INCH = 25.4
_CM_TO_INCH = 2.54


def autoscale_parts(parts):
    """Detect unit mismatches and ``_dN`` diameter targets and rescale parts.

    Strategy
    --------
    1. Use *outer* parts as the reference unit system.
    2. For each section (level), compare the inner (or mid) part diameter to
       the outer part diameter.  If the inner is much smaller than expected
       it was likely modelled in mm while the outer is in inches — scale
       it up by 25.4.  A factor around 2.54 indicates cm-to-inch.
    3. If the part name contains ``_dN`` (e.g. ``inner_2_3_d8``), uniformly
       scale the part so its XY diameter equals *N* in the outer-part units.

    Parameters
    ----------
    parts : list[(cq.Workplane, str)]
        Loaded parts (workplane + filename stem).

    Returns
    -------
    list[(cq.Workplane, str)]
        Parts with geometry scaled in-place where needed.
    """
    # Classify parts by tier and level so we can compare within sections.
    by_level = {}  # level -> {tier -> [(index, wp, name)]}
    for i, entry in enumerate(parts):
        wp, name = entry[0], entry[1]
        tier, levels, _seg = parse_part_name(name)
        if tier is None or levels is None:
            continue
        for lv in levels:
            by_level.setdefault(lv, {}).setdefault(tier, []).append(
                (i, wp, name)
            )

    # Collect outer reference diameters per level.
    outer_diams = {}  # level -> diameter
    for lv, tier_map in by_level.items():
        if "outer" in tier_map:
            _idx, wp, _name = tier_map["outer"][0]
            outer_diams[lv] = _get_xy_diameter(wp.val().wrapped)

    # Build a single "reference outer diameter" (median of all levels).
    if outer_diams:
        ref_outer_diam = sorted(outer_diams.values())[len(outer_diams) // 2]
    else:
        ref_outer_diam = None

    scaled = list(parts)  # shallow copy; we'll replace entries that change

    for lv, tier_map in by_level.items():
        outer_diam = outer_diams.get(lv, ref_outer_diam)
        if outer_diam is None:
            continue

        for tier in ("mid", "inner"):
            if tier not in tier_map:
                continue
            for idx, wp, name in tier_map[tier]:
                shape = wp.val().wrapped

                # --- Check for explicit _dN target diameter ---
                target_diam = _parse_target_diameter(name)
                if target_diam is not None:
                    cur_diam = _get_xy_diameter(shape)
                    if cur_diam > 1e-9:
                        factor = target_diam / cur_diam
                        if abs(factor - 1.0) > 0.01:
                            new_shape = scale_shape(shape, factor)
                            new_wp = cq.Workplane("XY").newObject(
                                [cq.Shape(new_shape)]
                            )
                            extra = parts[idx][2:] if len(parts[idx]) > 2 else ()
                            scaled[idx] = (new_wp, name) + extra
                            print(
                                f"  Autoscale {name}: diameter "
                                f"{cur_diam:.2f} -> {target_diam:.2f} "
                                f"(x{factor:.4f})"
                            )
                    continue  # _dN takes precedence; skip unit heuristic

                # --- Heuristic: detect mm-vs-inch mismatch ---
                cur_diam = _get_xy_diameter(shape)
                if cur_diam < 1e-9:
                    continue

                ratio = outer_diam / cur_diam
                if ratio > _INNER_TINY_RATIO:
                    # Inner is suspiciously small.  Try mm->inch scale.
                    test_mm = cur_diam * _MM_TO_INCH
                    test_cm = cur_diam * _CM_TO_INCH
                    # Pick the scale that brings diameter closest to
                    # outer (but still smaller or equal).
                    best_factor = None
                    best_diff = float("inf")
                    for f, label in (
                        (_MM_TO_INCH, "mm->in"),
                        (_CM_TO_INCH, "cm->in"),
                    ):
                        candidate = cur_diam * f
                        diff = abs(outer_diam - candidate)
                        if diff < best_diff:
                            best_diff = diff
                            best_factor = f
                            best_label = label

                    if best_factor is not None and abs(best_factor - 1.0) > 0.01:
                        new_shape = scale_shape(shape, best_factor)
                        new_wp = cq.Workplane("XY").newObject(
                            [cq.Shape(new_shape)]
                        )
                        extra = parts[idx][2:] if len(parts[idx]) > 2 else ()
                        scaled[idx] = (new_wp, name) + extra
                        print(
                            f"  Autoscale {name}: {best_label} "
                            f"(x{best_factor:.2f}), diameter "
                            f"{cur_diam:.2f} -> {cur_diam * best_factor:.2f} "
                            f"(outer ref {outer_diam:.2f})"
                        )

    return scaled


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

    Each entry in *parts* is either ``(wp, name)`` or ``(wp, name, is_mesh)``.
    The *is_mesh* flag (default ``False``) is carried through so downstream
    cutting can choose native mesh or parametric operations.

    Returns a CadQuery Assembly and a list of
    (name, ocp_shape, location, rgb_tuple, segment, is_mesh) tuples.
    """
    # --- Classify parts by tier and levels ---
    classified = []  # (tier, levels_list, segment, wp, name, original_index, is_mesh)
    auto_level = 1

    for i, entry in enumerate(parts):
        if len(entry) == 3:
            wp, name, is_mesh = entry
        else:
            wp, name = entry
            is_mesh = False
        tier, levels, segment = parse_part_name(name)
        if tier is None:
            # Unrecognized naming — treat as sequential outer parts
            tier = "outer"
            levels = [auto_level]
            segment = None
            auto_level += 1
        classified.append((tier, levels, segment, wp, name, i, is_mesh))

    # --- Build the set of levels and group by (level, tier) ---
    # Single-level parts go into levels_map[level][tier] as a list.
    # Multi-level (spanning) parts are collected separately.
    levels_map = {}  # level -> {tier -> [(wp, name, original_index, segment, is_mesh), ...]}
    spanning_parts = []  # (tier, levels_list, segment, wp, name, original_index, is_mesh)

    for tier, levels, segment, wp, name, idx, is_mesh in classified:
        if len(levels) == 1:
            levels_map.setdefault(levels[0], {}).setdefault(tier, []).append(
                (wp, name, idx, segment, is_mesh)
            )
        else:
            spanning_parts.append((tier, levels, segment, wp, name, idx, is_mesh))
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

        # Phase 1: compute tight XY centers before Phase 2/3 Z computations.
        # Shapes from orient_to_cylinder are fresh copies (BRepBuilderAPI_Transform
        # with copy=True in Step 4) that have no mesh yet.  The exact bbox on an
        # unmeshed shape is computed from the analytical parametric surfaces, which
        # can be inflated by degenerate BSpline faces (e.g. outer_1 exact X center
        # is -17.7 mm while the tight/real X center is +0.9 mm).  Using the tight
        # bbox here avoids that inflation and gives the correct XY center.
        # After this phase all shapes in this level are meshed, so subsequent
        # get_tight_bounding_box calls in Phase 2/3 are fast (no-op remesh).
        tight_bbox = {}  # original_index -> (xmin, ymin, zmin, xmax, ymax, zmax)
        tight_xy = {}    # original_index -> (cx, cy)
        for t in ("outer", "mid", "inner"):
            if t not in tier_group:
                continue
            for _wp, _nm, _idx, _seg, _ism in tier_group[t]:
                _sh = _wp.val().wrapped
                _bbox = get_tight_bounding_box(_sh)
                _xn, _yn, _, _xx, _yx, _ = _bbox
                tight_bbox[_idx] = _bbox
                tight_xy[_idx] = ((_xn + _xx) / 2.0, (_yn + _yx) / 2.0)

        # Phase 2: compute level height from the reference part's tight bbox.
        _, _, ozn_tight, _, _, ozx_tight = tight_bbox[ref_entry[2]]
        outer_height = ozx_tight - ozn_tight

        # Phase 3: position every part using pre-computed exact XY and tight Z.
        for tier in ("outer", "mid", "inner"):
            if tier not in tier_group:
                continue
            for wp, name, idx, segment, is_mesh in tier_group[tier]:
                shape = wp.val().wrapped
                part_cx, part_cy = tight_xy[idx]
                _, _, pzn_tight, _, _, _ = tight_bbox[idx]

                dx = -part_cx  # center XY at origin
                dy = -part_cy
                dz = z_cursor - pzn_tight

                loc = Location(Vector(dx, dy, dz))
                cq_color, rgb = pick_color(name, idx)

                assy.add(wp, name=name, loc=loc, color=cq_color)
                part_info.append((name, shape, loc, rgb, segment, is_mesh))

        # Advance the Z cursor past the outer part (+ gap)
        level_z_top[level] = z_cursor + outer_height
        z_cursor += outer_height + gap

    # --- Place spanning parts (parts that cover multiple levels) ---
    for tier, levels, segment, wp, name, idx, is_mesh in spanning_parts:
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
        part_info.append((name, shape, loc, rgb, segment, is_mesh))

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


# -------------------------------------------------------------------
# Native mesh direct-cut  (trimesh boolean, no OCP Splitter)
# -------------------------------------------------------------------

def _make_wedge_trimesh(cut_angle, axis_vec, origin_pt, radius, height):
    """Build a triangulated wedge sector mesh for boolean subtraction.

    The wedge covers the angular sector ``[0, cut_angle]`` around *axis_vec*
    through *origin_pt*, with the given *radius* and *height*.

    Returns a watertight ``trimesh.Trimesh`` representing the wedge solid.
    """
    import trimesh

    ox, oy, oz = origin_pt.X(), origin_pt.Y(), origin_pt.Z()
    cut_rad = math.radians(cut_angle)

    # Generate sector vertices in the local perpendicular plane
    n_steps = max(int(cut_angle / 3.0), 8)  # ~3° per step
    angles = np.linspace(0, cut_rad, n_steps + 1)

    # Perpendicular basis vectors
    if axis_vec.z:
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        ax = np.array([0.0, 0.0, 1.0])
    elif axis_vec.y:
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 1.0])
        ax = np.array([0.0, 1.0, 0.0])
    else:
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 1.0])
        ax = np.array([1.0, 0.0, 0.0])

    origin = np.array([ox, oy, oz])

    # Build sector polygon (origin + arc points) on the bottom and top
    half_h = height / 2.0
    bottom_center = origin - ax * half_h
    top_center = origin + ax * half_h

    # Bottom ring: origin, then arc points at radius
    n_arc = len(angles)  # n_steps + 1
    # Vertices: bottom_center, bottom_arc[0..n_arc-1],
    #           top_center,    top_arc[0..n_arc-1]
    verts = []
    # Bottom center (index 0)
    verts.append(bottom_center)
    # Bottom arc (indices 1..n_arc)
    for a in angles:
        pt = bottom_center + radius * (math.cos(a) * p1 + math.sin(a) * p2)
        verts.append(pt)
    # Top center (index n_arc + 1)
    verts.append(top_center)
    # Top arc (indices n_arc+2 .. 2*n_arc+1)
    for a in angles:
        pt = top_center + radius * (math.cos(a) * p1 + math.sin(a) * p2)
        verts.append(pt)

    verts = np.array(verts)
    bc = 0               # bottom center index
    ba = 1               # bottom arc start index
    tc = n_arc + 1        # top center index
    ta = n_arc + 2        # top arc start index

    faces = []
    for i in range(n_arc - 1):
        # Bottom fan triangle
        faces.append([bc, ba + i, ba + i + 1])
        # Top fan triangle (reversed winding)
        faces.append([tc, ta + i + 1, ta + i])
        # Side quad (two triangles)
        faces.append([ba + i, ta + i, ta + i + 1])
        faces.append([ba + i, ta + i + 1, ba + i + 1])

    # Two end-cap quads (at angle 0 and angle cut_angle)
    # Side at angle 0: (bc, ba+0, ta+0, tc)
    faces.append([bc, ta, ba])
    faces.append([bc, tc, ta])
    # Side at cut_angle: (bc, ba+n_arc-1, ta+n_arc-1, tc)
    last = n_arc - 1
    faces.append([bc, ba + last, ta + last])
    faces.append([bc, ta + last, tc])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    mesh.fix_normals()
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    return mesh


def _trimesh_to_ocp_solid(mesh):
    """Convert a trimesh.Trimesh to an OCP TopoDS_Shape (solid) via STL.

    Writes to a temporary STL file, loads with OCP, and converts to solid.
    Returns the OCP solid shape, or None on failure.
    """
    stl_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        stl_path = tmp.name
        tmp.close()
        mesh.export(stl_path)

        from OCP.StlAPI import StlAPI_Reader
        reader = StlAPI_Reader()
        shape = TopoDS_Shape()
        reader.Read(shape, stl_path)
        solid = _mesh_shell_to_solid(shape)
        return solid
    except Exception:
        return None
    finally:
        if stl_path and os.path.exists(stl_path):
            os.unlink(stl_path)


def _ensure_mesh_volume(mesh):
    """Ensure a trimesh mesh is a valid volume for boolean operations.

    Applies progressive repair strategies:
    1. Fill holes and fix normals on the whole mesh.
    2. If still not a volume, split into bodies and pick the largest volume.
    3. Returns the repaired mesh, or ``None`` if no valid body exists.
    """
    import trimesh as _tm

    if mesh.is_volume:
        return mesh

    _tm.repair.fill_holes(mesh)
    _tm.repair.fix_winding(mesh)
    _tm.repair.fix_normals(mesh)
    if mesh.is_volume:
        return mesh

    # Split into connected components and find the largest volume body
    try:
        bodies = mesh.split()
    except Exception:
        return None

    best = None
    best_vol = 0
    for body in bodies:
        if body.is_volume and abs(body.volume) > best_vol:
            best = body
            best_vol = abs(body.volume)

    return best


def cut_part_direct_mesh(shape, cut_angle, axis_vec, origin_pt):
    """Native mesh direct-cut: boolean subtraction using trimesh.

    Converts the OCP *shape* to a trimesh mesh, builds a wedge sector
    covering ``[0, cut_angle]``, performs a boolean difference in trimesh,
    and converts the result back to an OCP solid.

    This is the mesh-native counterpart of ``cut_part_direct()`` which uses
    the OCP ``BRepAlgoAPI_Splitter`` (best for parametric/B-rep geometry).

    Returns the kept OCP compound, or ``None`` if the result is empty.
    """
    import trimesh

    # Convert OCP shape to trimesh
    model_mesh = _shape_to_clean_trimesh(shape, tolerance=0.1)
    if model_mesh is None or len(model_mesh.faces) == 0:
        return None

    # Ensure the mesh is a volume for boolean operations.
    model_mesh = _ensure_mesh_volume(model_mesh)
    if model_mesh is None:
        return None

    # Compute bounding dimensions for the wedge cutter
    bounds = model_mesh.bounds  # (2, 3) min/max
    diag = np.linalg.norm(bounds[1] - bounds[0])
    radius = diag * 1.5
    height = diag * 3.0

    # Build the wedge mesh
    wedge = _make_wedge_trimesh(cut_angle, axis_vec, origin_pt, radius, height)

    # Boolean difference: model - wedge
    try:
        result_mesh = trimesh.boolean.difference([model_mesh, wedge], engine="manifold")
    except Exception:
        try:
            result_mesh = trimesh.boolean.difference([model_mesh, wedge], engine="blender")
        except Exception:
            return None

    if result_mesh is None or len(result_mesh.faces) == 0:
        return None

    # Convert back to OCP solid
    solid = _trimesh_to_ocp_solid(result_mesh)
    if solid is None or solid.IsNull():
        return None

    # Wrap in compound for consistency with cut_part_direct()
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, solid)
    return compound


def cut_part_direct_mesh_segment(shape, cut_angle, axis_vec, origin_pt, segment):
    """Native mesh direct-cut with segment splitting (a/b halves).

    After removing the main wedge ``[0, cut_angle]``, the remaining arc
    ``[cut_angle, 360]`` is bisected:

    * **segment "a"** keeps ``[cut_angle, midpoint]``
    * **segment "b"** keeps ``[midpoint, 360]``

    This is the mesh-native counterpart of ``cut_part_direct_segment()``.
    """
    import trimesh

    model_mesh = _shape_to_clean_trimesh(shape, tolerance=0.1)
    if model_mesh is None or len(model_mesh.faces) == 0:
        return None

    # Ensure the mesh is a volume for boolean operations.
    model_mesh = _ensure_mesh_volume(model_mesh)
    if model_mesh is None:
        return None

    bounds = model_mesh.bounds
    diag = np.linalg.norm(bounds[1] - bounds[0])
    radius = diag * 1.5
    height = diag * 3.0

    # Main wedge [0, cut_angle]
    wedge_main = _make_wedge_trimesh(cut_angle, axis_vec, origin_pt, radius, height)

    remaining = 360.0 - cut_angle
    midpoint = cut_angle + remaining / 2.0

    # Additional wedge to bisect the remaining arc
    if segment == "a":
        # Keep [cut_angle, midpoint] => also remove [midpoint, 360]
        remove_angle = 360.0 - midpoint
        wedge_extra = _make_wedge_trimesh(remove_angle, axis_vec, origin_pt, radius, height)
        # Rotate wedge_extra to start at midpoint
        _rotate_wedge = _make_wedge_trimesh(360.0 - midpoint, axis_vec, origin_pt, radius, height)
        # Actually: remove [0, cut_angle] and [midpoint, 360]
        # Simpler: remove [0, midpoint] entirely, then remove [0, cut_angle] from the complement
        # Better approach: build a wedge for [midpoint, 360] and union with main wedge
        wedge_seg = _make_wedge_trimesh(360.0 - midpoint, axis_vec, origin_pt, radius, height)
        # Rotate by midpoint degrees
        _rot_rad = math.radians(midpoint)
        if axis_vec.z:
            rot_axis = [0, 0, 1]
        elif axis_vec.y:
            rot_axis = [0, 1, 0]
        else:
            rot_axis = [1, 0, 0]
        rot_matrix = trimesh.transformations.rotation_matrix(_rot_rad, rot_axis,
                                                              [origin_pt.X(), origin_pt.Y(), origin_pt.Z()])
        wedge_seg.apply_transform(rot_matrix)
        # Union: remove both main wedge [0, cut_angle] and [midpoint, 360]
        try:
            combined_cutter = trimesh.boolean.union([wedge_main, wedge_seg], engine="manifold")
        except Exception:
            try:
                combined_cutter = trimesh.boolean.union([wedge_main, wedge_seg], engine="blender")
            except Exception:
                return None
    else:
        # segment "b": keep [midpoint, 360] => also remove [cut_angle, midpoint]
        # Remove [0, midpoint] entirely
        wedge_combined = _make_wedge_trimesh(midpoint, axis_vec, origin_pt, radius, height)
        combined_cutter = wedge_combined

    try:
        result_mesh = trimesh.boolean.difference([model_mesh, combined_cutter], engine="blender")
    except Exception:
        try:
            result_mesh = trimesh.boolean.difference([model_mesh, combined_cutter], engine="manifold")
        except Exception:
            return None

    if result_mesh is None or len(result_mesh.faces) == 0:
        return None

    solid = _trimesh_to_ocp_solid(result_mesh)
    if solid is None or solid.IsNull():
        return None

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, solid)
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
    bodies = []  # (name, mesh, offset[3], ocp_shape, loc, rgb, segment, is_mesh)
    for entry in part_info:
        if len(entry) == 6:
            name, shape, loc, rgb, segment, is_mesh = entry
        else:
            name, shape, loc, rgb, segment = entry
            is_mesh = False
        moved = apply_location(shape, loc)
        mesh = _shape_to_clean_trimesh(moved)
        if mesh is None:
            bodies.append((name, None, np.zeros(3), shape, loc, rgb, segment, is_mesh))
            continue
        hull = _mesh_convex_hull(mesh)
        bodies.append((name, mesh, np.zeros(3), shape, loc, rgb, segment, is_mesh))

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
        name, mesh, offset, shape, loc, rgb, segment, is_mesh = bodies[idx]
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
            bodies[idx] = (name, mesh, offset_new, shape, loc, rgb, segment, is_mesh)

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
        name, mesh, offset, shape, loc, rgb, segment, is_mesh = body
        if mesh is None:
            updated.append((name, shape, loc, rgb, segment, is_mesh))
        else:
            dx, dy, dz = offset[0], offset[1], offset[2]
            if abs(dx) < 1e-8 and abs(dy) < 1e-8 and abs(dz) < 1e-8:
                updated.append((name, shape, loc, rgb, segment, is_mesh))
            else:
                # Compose: apply original loc, then translate by sim offset
                new_loc = Location(Vector(dx, dy, dz)) * loc
                updated.append((name, shape, new_loc, rgb, segment, is_mesh))

    # -- Report bbox gap changes --
    _report_nesting(updated, ax, gap, debug)

    return updated


def _find_support_drop(falling_mesh, settled_mesh, ax):
    """Find how far *falling_mesh* can drop along *ax* before resting on
    *settled_mesh*.

    Uses two complementary methods and returns the most restrictive result:

    1. **Ray-casting** — a grid of downward rays from the falling mesh's
       bottom face detects horizontal support surfaces on the settled mesh.
       Works well for solid parts stacking on flat surfaces.

    2. **Collision binary-search** — samples vertices from the falling mesh
       and binary-searches for the maximum drop before any vertex enters the
       settled mesh's solid volume (via ``trimesh.contains``).  This catches
       radial / lateral collisions that vertical rays miss — e.g. when a
       smaller-diameter tube slides into a larger-diameter tube.

    Returns the maximum safe drop distance, or ``None`` if there is no
    interaction.
    """
    bounds_f = falling_mesh.bounds
    bounds_s = settled_mesh.bounds

    # ------------------------------------------------------------------
    # Method 1: Ray-casting (surface-to-surface support)
    # ------------------------------------------------------------------
    ray_drop = _find_support_drop_raycast(falling_mesh, settled_mesh, ax)

    # ------------------------------------------------------------------
    # Method 2: Collision binary-search (volume-to-volume penetration)
    # ------------------------------------------------------------------
    collision_drop = _find_support_drop_collision(
        falling_mesh, settled_mesh, ax,
    )

    # Take the most restrictive (smallest) drop from either method.
    if ray_drop is not None and collision_drop is not None:
        return min(ray_drop, collision_drop)
    if ray_drop is not None:
        return ray_drop
    if collision_drop is not None:
        return collision_drop
    return None


def _find_support_drop_raycast(falling_mesh, settled_mesh, ax):
    """Ray-cast method: grid of downward rays from the falling mesh bottom.

    Returns drop distance or ``None``.
    """
    bounds_f = falling_mesh.bounds

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

    try:
        hits, ray_ids, _ = settled_mesh.ray.intersects_location(
            origins, directions, multiple_hits=True
        )
    except Exception:
        return None

    if len(hits) == 0:
        return None

    ray_support = {}
    for hit, rid in zip(hits, ray_ids):
        hit_val = hit[ax]
        if rid not in ray_support or hit_val > ray_support[rid]:
            ray_support[rid] = hit_val

    if not ray_support:
        return None

    support_level = max(ray_support.values())
    drop = bounds_f[0, ax] - support_level

    if drop > 0:
        return drop

    return 0.0


def _find_support_drop_collision(falling_mesh, settled_mesh, ax,
                                 max_points=500, iterations=30,
                                 tolerance=0.05):
    """Binary-search for the maximum drop before the falling mesh's vertices
    penetrate the settled mesh's solid volume.

    Uses ``trimesh.Trimesh.contains()`` which checks whether points lie
    inside a watertight mesh volume.  For a hollow tube this correctly
    identifies points inside the *wall material* (not the hollow center).

    Returns drop distance or ``None`` if no collision is possible (meshes
    don't overlap laterally or settled mesh isn't watertight).
    """
    if not settled_mesh.is_watertight:
        return None

    bounds_f = falling_mesh.bounds
    bounds_s = settled_mesh.bounds

    # Quick lateral-overlap check (non-axis AABB).
    other_axes = [d for d in range(3) if d != ax]
    for oa in other_axes:
        if bounds_f[0, oa] >= bounds_s[1, oa] or \
           bounds_f[1, oa] <= bounds_s[0, oa]:
            return None

    # Maximum meaningful drop: falling mesh bottom → settled mesh bottom.
    max_drop = bounds_f[0, ax] - bounds_s[0, ax]
    if max_drop <= tolerance:
        return None

    # Subsample falling mesh vertices for speed.
    verts = falling_mesh.vertices
    if len(verts) > max_points:
        step = max(1, len(verts) // max_points)
        verts = verts[::step]

    # Binary search: find the largest drop with no penetration.
    lo = 0.0   # known safe (no drop)
    hi = max_drop

    for _ in range(iterations):
        mid = (lo + hi) * 0.5
        shifted = verts.copy()
        shifted[:, ax] -= mid
        try:
            inside = settled_mesh.contains(shifted)
        except Exception:
            return None
        if inside.any():
            hi = mid  # collision — drop less
        else:
            lo = mid  # safe — try dropping more

    # Only return a result if the collision limit is meaningfully less than
    # the full drop (otherwise the ray-cast method is better).
    if hi < max_drop - tolerance:
        return lo

    return None


def _report_nesting(part_info, ax, original_gap, debug):
    """Print nesting report showing effective gaps vs bounding-box gaps."""
    if not part_info:
        return

    # Compute bounding boxes of final positioned parts
    positioned = []
    for entry in part_info:
        name, shape, loc, rgb, segment = entry[:5]
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
    assy.export(filepath, "STEP")


def export_shape_step(shape, filepath):
    """Export a raw OCP shape to STEP."""
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filepath)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed with status {status}")


def _is_step_path(filepath):
    return os.path.splitext(filepath)[1].lower() in {".step", ".stp"}


def export_shape(shape, filepath):
    """Export a raw OCP shape to STEP/STL based on output extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in {".step", ".stp"}:
        export_shape_step(shape, filepath)
        return
    if ext == ".stl":
        cq.exporters.export(cq.Shape.cast(shape), filepath, exportType="STL")
        return
    raise ValueError(
        f"Unsupported output format '{ext}'. Use .step/.stp or .stl for --output."
    )




def _faces_to_triangles(faces):
    """Convert tessellation face records to an (N, 3) triangle index array."""
    tris = []
    for face in faces:
        row = list(face)
        if not row:
            continue
        n = int(row[0])
        if n < 3:
            continue
        for i in range(1, n - 1):
            tris.append([int(row[1]), int(row[i + 1]), int(row[i + 2])])
    return np.array(tris, dtype=np.int64) if tris else np.empty((0, 3), dtype=np.int64)


def export_part_native(shape, output_path, source_ext, is_mesh):
    """Export a transformed part using the original file format where possible."""
    ext = source_ext.lower()

    if not is_mesh:
        if ext in {".step", ".stp"}:
            export_shape_step(shape, output_path)
            return
        if ext == ".brep":
            BRepTools.Write_s(shape, output_path)
            return
        if ext in {".iges", ".igs"}:
            from OCP.IGESControl import IGESControl_Writer
            writer = IGESControl_Writer()
            writer.AddShape(shape)
            if not writer.Write(output_path):
                raise RuntimeError(f"IGES write failed: {output_path}")
            return
        # fallback for unsupported CAD writers
        export_shape_step(shape, output_path)
        return

    # Mesh export path: tessellate once and write using trimesh to native mesh format
    import trimesh

    verts, faces = tessellate_shape(shape, tolerance=0.2, angular=0.5)
    tri = _faces_to_triangles(faces)
    if len(verts) == 0 or len(tri) == 0:
        raise RuntimeError(f"Unable to tessellate mesh part for export: {output_path}")

    mesh = trimesh.Trimesh(vertices=verts, faces=tri, process=False)
    mesh.export(output_path)


def export_transformed_parts(parts, output_dir):
    """Write pre-stacking transformed parts to *output_dir* in native formats."""
    os.makedirs(output_dir, exist_ok=True)
    used_names = {}

    print(f"\nExporting transformed parts (--parts): {output_dir}")
    for part in parts:
        name = part["name"]
        shape = part["shape"]
        src_ext = part["source_ext"]
        is_mesh = part["is_mesh"]

        suffix = used_names.get(name, 0)
        used_names[name] = suffix + 1
        out_name = f"{name}_{suffix}{src_ext}" if suffix else f"{name}{src_ext}"

        out_ext = src_ext.lower()
        if (not is_mesh) and out_ext not in {".step", ".stp", ".iges", ".igs", ".brep"}:
            out_name = os.path.splitext(out_name)[0] + ".step"

        out_path = os.path.join(output_dir, out_name)
        export_part_native(shape, out_path, os.path.splitext(out_name)[1], is_mesh)
        print(f"  Wrote: {out_path}")

def build_moved_compound(part_info):
    """Build a compound from positioned parts in part_info."""
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for entry in part_info:
        name, shape, loc = entry[0], entry[1], entry[2]
        moved = apply_location(shape, loc)
        builder.Add(compound, moved)
    return compound


def _scale_shape_about_center(shape, clearance):
    """Return *shape* scaled outward about its bbox center by *clearance*."""
    xn, yn, zn, xx, yx, zx = get_bounding_box(shape)
    cx = (xn + xx) / 2.0
    cy = (yn + yx) / 2.0
    cz = (zn + zx) / 2.0
    max_dim = max(xx - xn, yx - yn, zx - zn)
    if max_dim <= 1e-9:
        return shape

    factor = (max_dim + 2.0 * clearance) / max_dim
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(cx, cy, cz), factor)
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _scale_shape_anisotropic_about_center(shape, sx=1.0, sy=1.0, sz=1.0):
    """Return *shape* scaled about bbox center with independent XYZ factors."""
    xn, yn, zn, xx, yx, zx = get_bounding_box(shape)
    cx = (xn + xx) / 2.0
    cy = (yn + yx) / 2.0
    cz = (zn + zx) / 2.0

    gtrsf = gp_GTrsf()
    gtrsf.SetValue(1, 1, sx)
    gtrsf.SetValue(1, 2, 0.0)
    gtrsf.SetValue(1, 3, 0.0)
    gtrsf.SetValue(1, 4, cx * (1.0 - sx))

    gtrsf.SetValue(2, 1, 0.0)
    gtrsf.SetValue(2, 2, sy)
    gtrsf.SetValue(2, 3, 0.0)
    gtrsf.SetValue(2, 4, cy * (1.0 - sy))

    gtrsf.SetValue(3, 1, 0.0)
    gtrsf.SetValue(3, 2, 0.0)
    gtrsf.SetValue(3, 3, sz)
    gtrsf.SetValue(3, 4, cz * (1.0 - sz))

    return BRepBuilderAPI_GTransform(shape, gtrsf, True).Shape()


def _shape_volume(shape):
    """Return shape volume (0 on failure)."""
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    try:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        return max(0.0, props.Mass())
    except Exception:
        return 0.0


def _has_positive_common_volume(shape_a, shape_b, tol=1e-7):
    """Return True when shapes have a non-trivial volume overlap."""
    common = BRepAlgoAPI_Common(shape_a, shape_b)
    common.Build()
    if not common.IsDone():
        return False
    return _shape_volume(common.Shape()) > tol


def midscale_parts(part_info, debug=False):
    """Expand mid-tier parts after stacking: XY to outer contact, then Z.

    XY scaling is uniform (single factor for X and Y). Z scaling uses an
    independent factor.
    """
    world_entries = []
    for entry in part_info:
        name, shape, loc, rgb = entry[0], entry[1], entry[2], entry[3]
        segment = entry[4] if len(entry) > 4 else None
        is_mesh = entry[5] if len(entry) > 5 else False
        tier, levels, _ = parse_part_name(name)
        world_entries.append({
            "name": name,
            "shape": apply_location(shape, loc),
            "rgb": rgb,
            "segment": segment,
            "is_mesh": is_mesh,
            "tier": tier,
            "levels": set(levels or []),
        })

    mids = [e for e in world_entries if e["tier"] == "mid"]
    outers = [e for e in world_entries if e["tier"] == "outer"]
    z_neighbors = [e for e in world_entries if e["tier"] in {"mid", "outer"}]
    changed = 0

    for mid in mids:
        shape = mid["shape"]

        # --- XY: uniform scale until non-trivial overlap with matching outer ---
        outer_targets = [
            o for o in outers
            if mid["levels"] and o["levels"] and (mid["levels"] & o["levels"])
        ]
        if outer_targets:
            target_outer = outer_targets[0]["shape"]
            if not _has_positive_common_volume(shape, target_outer):
                low, high = 1.0, 1.05
                found = False
                for _ in range(20):
                    trial = _scale_shape_anisotropic_about_center(shape, sx=high, sy=high, sz=1.0)
                    if _has_positive_common_volume(trial, target_outer):
                        found = True
                        break
                    low = high
                    high *= 1.25
                if found:
                    for _ in range(24):
                        midf = 0.5 * (low + high)
                        trial = _scale_shape_anisotropic_about_center(shape, sx=midf, sy=midf, sz=1.0)
                        if _has_positive_common_volume(trial, target_outer):
                            high = midf
                        else:
                            low = midf
                    if high > 1.0 + 1e-6:
                        shape = _scale_shape_anisotropic_about_center(shape, sx=high, sy=high, sz=1.0)
                        if debug:
                            print(f"  [DEBUG] Midscale XY {mid['name']}: x{high:.5f}")

        # --- Z: scale until contacting nearest mid/outer above and below ---
        mnx, mny, mnz, mxx, myy, mxz = get_bounding_box(shape)
        height = mxz - mnz
        if height > 1e-9:
            gap_above = None
            gap_below = None
            for other in z_neighbors:
                if other is mid:
                    continue
                onx, ony, onz, oxx, oyy, oxz = get_bounding_box(other["shape"])
                overlap_xy = not (oxx <= mnx or onx >= mxx or oyy <= mny or ony >= myy)
                if not overlap_xy:
                    continue
                if onz >= mxz:
                    g = onz - mxz
                    gap_above = g if gap_above is None else min(gap_above, g)
                elif oxz <= mnz:
                    g = mnz - oxz
                    gap_below = g if gap_below is None else min(gap_below, g)

            if gap_above is not None and gap_below is not None:
                delta = max(gap_above, gap_below)
                if delta > 1e-9:
                    zf = 1.0 + (2.0 * delta / height)
                    shape = _scale_shape_anisotropic_about_center(shape, sx=1.0, sy=1.0, sz=zf)
                    if debug:
                        print(
                            f"  [DEBUG] Midscale Z {mid['name']}: x{zf:.5f} "
                            f"(gaps up={gap_above:.5f}, down={gap_below:.5f})"
                        )

        if shape is not mid["shape"]:
            changed += 1
            mid["shape"] = shape

    updated = []
    for e in world_entries:
        updated.append((e["name"], e["shape"], Location(), e["rgb"], e["segment"], e["is_mesh"]))

    print(f"  Midscale complete ({changed} mid part(s) adjusted).")
    return updated


def mid_cut_parts(part_info, output_dir="parts", clearance=0.02, debug=False):
    """Cut mid-tier parts to clear inner-tier parts and export the cut mids.

    This operation is applied in world coordinates.  Returned part_info entries
    are emitted with identity locations because their geometry is already moved.
    """
    os.makedirs(output_dir, exist_ok=True)

    world_entries = []
    for entry in part_info:
        name, shape, loc, rgb = entry[0], entry[1], entry[2], entry[3]
        segment = entry[4] if len(entry) > 4 else None
        is_mesh = entry[5] if len(entry) > 5 else False
        tier, levels, _ = parse_part_name(name)
        moved = apply_location(shape, loc)
        world_entries.append({
            "name": name,
            "shape": moved,
            "rgb": rgb,
            "segment": segment,
            "is_mesh": is_mesh,
            "tier": tier,
            "levels": set(levels or []),
        })

    inner_entries = [e for e in world_entries if e["tier"] == "inner"]
    cut_count = 0

    for entry in world_entries:
        if entry["tier"] != "mid":
            continue

        matching_inners = [
            inner for inner in inner_entries
            if entry["levels"] and inner["levels"] and entry["levels"] & inner["levels"]
        ]
        if not matching_inners:
            continue

        cutter_builder = BRep_Builder()
        cutter = TopoDS_Compound()
        cutter_builder.MakeCompound(cutter)
        for inner in matching_inners:
            inflated = _scale_shape_about_center(inner["shape"], clearance)
            cutter_builder.Add(cutter, inflated)

        cut_mid = cut_assembly(entry["shape"], cutter)
        if cut_mid is not None:
            entry["shape"] = cut_mid
            cut_count += 1
            out_path = os.path.join(output_dir, f"{entry['name']}_mid_cut.step")
            export_part_native(cut_mid, out_path, ".step", is_mesh=False)
            if debug:
                print(f"  [DEBUG] Mid cut exported: {out_path}")

    updated = []
    for e in world_entries:
        updated.append((e["name"], e["shape"], Location(), e["rgb"], e["segment"], e["is_mesh"]))

    print(f"  Mid-cut complete ({cut_count} mid part(s) cut, clearance={clearance:.4f} in).")
    print(f"  Mid-cut parts saved in: {output_dir}")
    return updated


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

    # Camera (slightly zoomed out so tall parts are less likely to clip top/bottom)
    plotter.camera.zoom(0.9)

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


def cut_shape_by_plane(shape, plane_axis="y", plane_value=0.0, keep_negative=True):
    """Cut *shape* by an axis-aligned plane and keep one side.

    keep_negative=True keeps the <= plane side.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = get_bounding_box(shape)
    x_span = max(1e-6, xmax - xmin)
    y_span = max(1e-6, ymax - ymin)
    z_span = max(1e-6, zmax - zmin)
    pad = max(x_span, y_span, z_span) * 2.0

    box_dx = x_span + 2 * pad
    box_dy = y_span + 2 * pad
    box_dz = z_span + 2 * pad

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    cz = (zmin + zmax) * 0.5

    if plane_axis == "x":
        cut_center_x = plane_value + (box_dx * 0.5 if keep_negative else -box_dx * 0.5)
        cutter = cq.Workplane("XY").box(box_dx, box_dy, box_dz).translate((cut_center_x, cy, cz))
    elif plane_axis == "y":
        cut_center_y = plane_value + (box_dy * 0.5 if keep_negative else -box_dy * 0.5)
        cutter = cq.Workplane("XY").box(box_dx, box_dy, box_dz).translate((cx, cut_center_y, cz))
    elif plane_axis == "z":
        cut_center_z = plane_value + (box_dz * 0.5 if keep_negative else -box_dz * 0.5)
        cutter = cq.Workplane("XY").box(box_dx, box_dy, box_dz).translate((cx, cy, cut_center_z))
    else:
        raise ValueError(f"Unsupported plane_axis '{plane_axis}'. Use x/y/z.")

    return cut_assembly(shape, cutter.val().wrapped)


def _render_topdown_mask(shape, reference_bounds, resolution=512):
    """Render a deterministic top-down binary mask for validation."""
    import pyvista as pv

    verts, faces = tessellate_shape(shape, tolerance=0.05)
    if len(verts) == 0:
        return np.zeros((resolution, resolution), dtype=bool)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=[resolution, resolution])
    mesh = pv.PolyData(verts, faces.ravel())
    plotter.add_mesh(mesh, color=(1, 1, 1), lighting=False)
    plotter.set_background("black")

    xmin, ymin, zmin, xmax, ymax, zmax = reference_bounds
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    cz = (zmin + zmax) * 0.5
    span_xy = max(xmax - xmin, ymax - ymin)
    span_z = max(1e-6, zmax - zmin)

    cam = plotter.camera
    cam.parallel_projection = True
    cam.parallel_scale = max(1e-6, span_xy * 0.52)
    cam.position = (cx, cy, zmax + 4.0 * span_z + 1.0)
    cam.focal_point = (cx, cy, cz)
    cam.up = (0, 1, 0)

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img[..., :3].max(axis=2) > 0


def validate_planar_cut_projection(shape, plane_axis="y", plane_value=0.0,
                                   keep_negative=True, resolution=512,
                                   max_mismatch_ratio=0.01):
    """Validate that a 3D half-space cut matches a 2D post-render half-image mask."""
    bounds = get_bounding_box(shape)
    cut_shape = cut_shape_by_plane(
        shape, plane_axis=plane_axis, plane_value=plane_value, keep_negative=keep_negative,
    )
    if cut_shape is None:
        return False, 1.0, "Cut removed entire shape."

    cut_mask = _render_topdown_mask(cut_shape, bounds, resolution=resolution)
    full_mask = _render_topdown_mask(shape, bounds, resolution=resolution)
    expected_mask = full_mask.copy()

    xmin, ymin, _zmin, xmax, ymax, _zmax = bounds
    if plane_axis == "y":
        denom = max(1e-6, ymax - ymin)
        row_plane = int(round((ymax - plane_value) / denom * (resolution - 1)))
        row_plane = max(0, min(resolution, row_plane))
        if keep_negative:
            expected_mask[:row_plane, :] = False
        else:
            expected_mask[row_plane:, :] = False
    elif plane_axis == "x":
        denom = max(1e-6, xmax - xmin)
        col_plane = int(round((plane_value - xmin) / denom * (resolution - 1)))
        col_plane = max(0, min(resolution, col_plane))
        if keep_negative:
            expected_mask[:, col_plane:] = False
        else:
            expected_mask[:, :col_plane] = False
    else:
        return False, 1.0, "Top-down projection validation supports x/y planes only."

    mismatch = np.logical_xor(cut_mask, expected_mask).sum()
    norm = max(1, expected_mask.sum())
    mismatch_ratio = mismatch / norm
    if mismatch_ratio <= max_mismatch_ratio:
        return True, mismatch_ratio, "ok"

    return False, mismatch_ratio, (
        f"Mismatch ratio {mismatch_ratio:.4f} exceeds limit {max_mismatch_ratio:.4f}."
    )


def run_validation_pipeline(shape, resolution=512, mismatch_ratio=0.01, debug=False):
    """Run validation checks intended for automated tests and CLI --validate."""
    checks = []
    ok, ratio, msg = validate_planar_cut_projection(
        shape,
        plane_axis="y",
        plane_value=0.0,
        keep_negative=True,
        resolution=resolution,
        max_mismatch_ratio=mismatch_ratio,
    )
    checks.append({
        "name": "plane_cut_projection_y0_topdown",
        "ok": ok,
        "mismatch_ratio": ratio,
        "message": msg,
    })

    all_ok = all(c["ok"] for c in checks)
    if debug:
        for c in checks:
            state = "PASS" if c["ok"] else "FAIL"
            print(f"  [VALIDATE] {state} {c['name']}: {c['message']} (mismatch={c['mismatch_ratio']:.6f})")

    return all_ok, checks


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
    loaded_parts_for_export = []
    # FSTL-style fast path: keep mesh as triangle shells during load and avoid
    # expensive mesh->solid sewing unless a downstream operation truly needs it.
    require_mesh_solids = False
    max_workers = min(len(valid_paths), max(1, min(8, (os.cpu_count() or 4))))

    # Deduplicate loads by realpath to avoid re-reading identical CAD files.
    # This commonly happens when users pass overlapping glob patterns.
    unique_by_real = {}
    for fp in valid_paths:
        unique_by_real.setdefault(os.path.realpath(fp), fp)

    if len(unique_by_real) > 1:
        # Parallel loading for unique files only.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(load_part, fp, require_solid=require_mesh_solids): real
                for real, fp in unique_by_real.items()
            }
            loaded_unique = {}
            for future in as_completed(future_map):
                real = future_map[future]
                src_fp = unique_by_real[real]
                try:
                    wp, _name = future.result()
                    loaded_unique[real] = wp
                    print(f"  Loaded: {os.path.splitext(os.path.basename(src_fp))[0]}")
                except Exception as e:
                    print(f"  ERROR loading '{src_fp}': {e}")
    else:
        loaded_unique = {}
        for real, filepath in unique_by_real.items():
            try:
                wp, _name = load_part(filepath, require_solid=require_mesh_solids)
                loaded_unique[real] = wp
                print(f"  Loaded: {os.path.splitext(os.path.basename(filepath))[0]}")
            except Exception as e:
                print(f"  ERROR loading '{filepath}': {e}")

    # Preserve original input order, but reuse already loaded geometry.
    for fp in valid_paths:
        real = os.path.realpath(fp)
        wp = loaded_unique.get(real)
        if wp is None:
            continue
        name = os.path.splitext(os.path.basename(fp))[0]
        mesh_flag = is_mesh_file(fp)
        parts.append((wp, name, mesh_flag))
        loaded_parts_for_export.append({
            "name": name,
            "shape": wp.val().wrapped,
            "is_mesh": mesh_flag,
            "source_ext": os.path.splitext(fp)[1],
        })

    if not parts:
        print("No valid parts loaded. Exiting.")
        return 1

    # 1a. Autoscale (optional) — runs before orient so diameters are correct
    if getattr(args, "autoscale", False):
        print("\nAuto-scaling parts (--autoscale)...")
        parts = autoscale_parts(parts)

    # 1b. Cylinder orientation (optional)
    if getattr(args, "cyl", False):
        print("\nOrient parts for cylinder (--cyl)...")
        parts = orient_to_cylinder(parts, gap=args.gap)
        # orient_to_cylinder creates transformed copies; refresh export geometry.
        for i, entry in enumerate(parts):
            loaded_parts_for_export[i]["shape"] = entry[0].val().wrapped

    if getattr(args, "parts", None):
        export_transformed_parts(loaded_parts_for_export, args.parts)

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
        for entry in part_info:
            name, shape, loc, rgb = entry[0], entry[1], entry[2], entry[3]
            wp = cq.Workplane("XY").newObject([cq.Shape(shape)])
            cq_color = Color(*rgb) if len(rgb) == 3 else Color(*rgb[:3])
            assy.add(wp, name=name, loc=loc, color=cq_color)
        print("  Physics simulation complete.")

    # 2c. Mid-scale (optional): expand mid-tier parts to contact neighbors
    if getattr(args, "midscale", False):
        print("\nApplying mid scaling (--midscale)...")
        part_info = midscale_parts(
            part_info,
            debug=getattr(args, "debug", False),
        )
        assy = Assembly()
        for entry in part_info:
            name, shape, loc, rgb = entry[0], entry[1], entry[2], entry[3]
            wp = cq.Workplane("XY").newObject([cq.Shape(shape)])
            cq_color = Color(*rgb) if len(rgb) == 3 else Color(*rgb[:3])
            assy.add(wp, name=name, loc=loc, color=cq_color)

    # 2d. Mid-cut (optional): hollow mid-tier parts using inner-tier clearance
    if getattr(args, "mid_cut", False):
        print("\nApplying mid-part clearance cuts (--mid_cut)...")
        part_info = mid_cut_parts(
            part_info,
            output_dir=(args.parts if getattr(args, "parts", None) else "parts"),
            clearance=0.02,
            debug=getattr(args, "debug", False),
        )
        assy = Assembly()
        for entry in part_info:
            name, shape, loc, rgb = entry[0], entry[1], entry[2], entry[3]
            wp = cq.Workplane("XY").newObject([cq.Shape(shape)])
            cq_color = Color(*rgb) if len(rgb) == 3 else Color(*rgb[:3])
            assy.add(wp, name=name, loc=loc, color=cq_color)

    # 3. Cut (optional) and export output file
    output_path = args.output
    cut_result_shape = None
    moved_compound = build_moved_compound(part_info)

    if args.cut_angle is not None:
        # Export pre-cut assembly
        precut_path = os.path.splitext(output_path)[0] + "_precut.step"
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
        for entry in part_info:
            name, shape, loc = entry[0], entry[1], entry[2]
            segment = entry[4]
            is_mesh = entry[5] if len(entry) > 5 else False
            moved = apply_location(shape, loc)
            builder.Add(compound, moved)
            moved_parts.append((name, moved, segment, is_mesh))

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
        has_segments = any(seg is not None for _, _, seg, _ in moved_parts)

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

        for pname, moved_shape, segment, part_is_mesh in moved_parts:
            try:
                if use_direct:
                    # --- Direct geometry approach ---
                    # Choose native operation based on model type:
                    # mesh-origin parts use trimesh booleans,
                    # parametric/B-rep parts use OCP Splitter.
                    if part_is_mesh:
                        if segment is not None:
                            cut_part = cut_part_direct_mesh_segment(
                                moved_shape, args.cut_angle, axis_vec,
                                origin_pt, segment,
                            )
                        else:
                            cut_part = cut_part_direct_mesh(
                                moved_shape, args.cut_angle, axis_vec,
                                origin_pt,
                            )
                    elif segment is not None:
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

        print(f"Exporting cut assembly: {output_path}")
        export_shape(cut_result_shape, output_path)
    else:
        print(f"\nExporting assembly: {output_path}")
        if _is_step_path(output_path):
            try:
                export_assembly_step(assy, output_path)
            except Exception as e:
                print(f"  Assembly export fallback: {e}")
                export_shape(moved_compound, output_path)
        else:
            export_shape(moved_compound, output_path)

    print(f"  Output saved: {output_path}")

    # 4. Validate (optional)
    if getattr(args, "validate", False):
        print("\nRunning validation checks (--validate)...")
        validate_target = cut_result_shape if cut_result_shape is not None else moved_compound
        valid_ok, valid_checks = run_validation_pipeline(
            validate_target,
            resolution=getattr(args, "validate_resolution", 512),
            mismatch_ratio=getattr(args, "validate_max_mismatch", 0.01),
            debug=getattr(args, "debug", False),
        )
        for check in valid_checks:
            status = "PASS" if check["ok"] else "FAIL"
            print(
                f"  [{status}] {check['name']} | mismatch={check['mismatch_ratio']:.6f} | {check['message']}"
            )
        if not valid_ok:
            print("Validation failed.")
            return 1

    # 5. Render
    if args.render:
        print(f"\nRendering to {args.render} ({args.resolution}x{args.resolution})...")
        render_assembly(
            part_info, args.render,
            resolution=args.resolution,
            cut_shape=cut_result_shape,
        )

    # 6. Summary
    print("\n=== Pipeline Complete ===")
    print(f"  Parts:      {len(parts)}")
    print(f"  Cyl orient: {getattr(args, 'cyl', False)}")
    print(f"  Physics:    {getattr(args, 'phys', False)}")
    print(f"  Axis:       {args.axis.upper()}")
    print(f"  Gap:        {args.gap}")
    if args.cut_angle is not None:
        cut_method = "direct" if getattr(args, "cut_direct", False) else "boolean"
        print(f"  Cut angle:  {args.cut_angle} deg ({cut_method})")
    print(f"  Output:     {output_path}")
    if args.render:
        print(f"  Render out: {args.render}")
    if getattr(args, "validate", False):
        print("  Validate:   enabled")

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
        "--autoscale", action="store_true",
        help="Auto-detect unit mismatches between parts and rescale.  "
             "Compares inner/mid diameters to outer parts; if an inner part "
             "is tiny relative to its outer, it is scaled up (mm->inch by "
             "25.4 or cm->inch by 2.54).  Parts with a '_dN' tag in the "
             "name (e.g. inner_2_3_d8) are scaled to diameter N in the "
             "outer-part unit system.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation pipeline comparing a 3D y=0 plane cut against a 2D masked top-down render.",
    )
    parser.add_argument(
        "--validate-resolution", type=int, default=512,
        help="Resolution used for validation renders (default: 512)",
    )
    parser.add_argument(
        "--validate-max-mismatch", type=float, default=0.01,
        help="Maximum normalized pixel mismatch tolerated by validation (default: 0.01)",
    )
    parser.add_argument(
        "--parts", nargs="?", const="parts", default=None,
        help="Export rotated/scaled pre-stack parts to a directory (default: ./parts).",
    )
    parser.add_argument(
        "--midscale", action="store_true",
        help="Scale mid-tier parts after stacking: uniform XY until contacting outer, then Z until contacting mid/outer above and below.",
    )
    parser.add_argument(
        "--mid_cut", action="store_true",
        help="Cut mid-tier parts to create 0.02 in clearance for matching inner-tier parts and export cut mids to ./parts (or --parts dir).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output",
    )
    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
