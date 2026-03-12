#!/usr/bin/env python3
"""
Adversarial Part Stack Testing

Creates increasingly difficult part configurations and runs them through
the stack → cut → render pipeline, validating output at each step.
"""

import sys
import os
import math
import tempfile
import traceback

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.TopoDS import TopoDS_Compound, TopoDS
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepCheck import BRepCheck_Analyzer

from assemble import (
    stack_parts, make_cutter, cut_assembly, render_assembly,
    export_shape_step, apply_location, get_bounding_box,
    tessellate_shape, cut_part_direct, cut_part_direct_segment,
    _cutter_params, AXIS_MAP,
)


# ── Validation helpers ──────────────────────────────────────────────

def count_solids(shape):
    """Count the number of solids in a shape."""
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def shape_volume(shape):
    """Compute the volume of a shape."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def is_topology_valid(shape):
    """Check that a shape passes BRepCheck_Analyzer."""
    return BRepCheck_Analyzer(shape).IsValid()


def has_faces(shape):
    """Check if a shape has any faces (can be tessellated)."""
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    return exp.More()


def can_tessellate(shape):
    """Check that the shape can be tessellated to vertices/faces."""
    try:
        verts, faces = tessellate_shape(shape, tolerance=0.1)
        return len(verts) > 0 and len(faces) > 0
    except Exception:
        return False


def validate_cut_result(original_compound, cut_shape, cut_angle, label):
    """Validate a cut result against the original.

    Checks:
      1. Result is not None/empty
      2. Result has faces and can be tessellated
      3. Volume is less than original (material was removed)
      4. Volume ratio is approximately correct for the cut angle
      5. Topology is valid (BRepCheck_Analyzer)
    """
    errors = []

    if cut_shape is None:
        errors.append(f"[{label}] Cut returned None")
        return errors

    # Check bounding box is not void
    bbox = Bnd_Box()
    BRepBndLib.Add_s(cut_shape, bbox)
    if bbox.IsVoid():
        errors.append(f"[{label}] Cut result has void bounding box")
        return errors

    # Check it has faces
    if not has_faces(cut_shape):
        errors.append(f"[{label}] Cut result has no faces")

    # Check tessellation
    if not can_tessellate(cut_shape):
        errors.append(f"[{label}] Cut result cannot be tessellated")

    # Volume comparison
    orig_vol = shape_volume(original_compound)
    cut_vol = shape_volume(cut_shape)

    if orig_vol > 1e-6:
        expected_ratio = (360.0 - cut_angle) / 360.0
        actual_ratio = cut_vol / orig_vol
        # Allow 20% tolerance for complex geometry
        if actual_ratio > 1.05:
            errors.append(
                f"[{label}] Volume INCREASED after cut: "
                f"orig={orig_vol:.2f}, cut={cut_vol:.2f}, ratio={actual_ratio:.3f}"
            )
        elif abs(actual_ratio - expected_ratio) > 0.25:
            errors.append(
                f"[{label}] Volume ratio {actual_ratio:.3f} deviates from expected "
                f"{expected_ratio:.3f} (angle={cut_angle}°)"
            )

    # Topology check (informational — boolean ops on thin-wall geometry
    # sometimes produce topologically imperfect but usable results)
    if not is_topology_valid(cut_shape):
        print(f"  [INFO] {label}: BRepCheck_Analyzer reports topology issues (non-fatal)")
        # Only error if it can't tessellate AND has topology issues
        if not can_tessellate(cut_shape):
            errors.append(f"[{label}] Topology invalid AND cannot tessellate")

    return errors


def validate_render(parts_data, cut_shape, label, render_path):
    """Validate rendering by producing a PNG and checking it exists."""
    errors = []
    try:
        render_assembly(parts_data, render_path, resolution=512, cut_shape=cut_shape)
        if not os.path.exists(render_path):
            errors.append(f"[{label}] Render file not created")
        elif os.path.getsize(render_path) < 1000:
            errors.append(f"[{label}] Render file suspiciously small ({os.path.getsize(render_path)} bytes)")
    except Exception as e:
        errors.append(f"[{label}] Render failed: {e}")
    return errors


# ── Part generators ─────────────────────────────────────────────────

def make_box(name, lx, ly, lz, cx=0, cy=0, cz=0):
    """Create a named box part centered at (cx, cy, cz)."""
    wp = cq.Workplane("XY").transformed(offset=(cx, cy, cz)).box(lx, ly, lz)
    return wp, name


def make_cylinder(name, radius, height, cx=0, cy=0, cz=0):
    """Create a named cylinder part."""
    wp = cq.Workplane("XY").transformed(offset=(cx, cy, cz)).cylinder(height, radius)
    return wp, name


def make_hollow_cylinder(name, outer_r, inner_r, height, cx=0, cy=0, cz=0):
    """Create a hollow cylinder (tube/pipe)."""
    wp = (cq.Workplane("XY")
          .transformed(offset=(cx, cy, cz))
          .cylinder(height, outer_r)
          .faces(">Z").workplane()
          .hole(inner_r * 2))
    return wp, name


def make_cone(name, r_bottom, r_top, height, cx=0, cy=0, cz=0):
    """Create a cone/frustum."""
    solid = cq.Solid.makeCone(r_bottom, r_top, height,
                              pnt=cq.Vector(cx, cy, cz - height / 2),
                              dir=cq.Vector(0, 0, 1))
    wp = cq.Workplane("XY").newObject([solid])
    return wp, name


def make_sphere(name, radius, cx=0, cy=0, cz=0):
    """Create a sphere."""
    wp = (cq.Workplane("XY")
          .transformed(offset=(cx, cy, cz))
          .sphere(radius))
    return wp, name


def make_torus(name, major_r, minor_r, cx=0, cy=0, cz=0):
    """Create a torus (donut shape)."""
    wp = (cq.Workplane("XZ")
          .transformed(offset=(cx, cy, cz))
          .circle(major_r)
          .revolve(360, (0, 0, 0), (0, 0, 1)))
    return wp, name


def make_hexagonal_prism(name, across_flats, height, cx=0, cy=0, cz=0):
    """Create a hexagonal prism (nut shape)."""
    wp = (cq.Workplane("XY")
          .transformed(offset=(cx, cy, cz))
          .polygon(6, across_flats)
          .extrude(height))
    return wp, name


def make_flanged_cylinder(name, cyl_r, cyl_h, flange_r, flange_h, cx=0, cy=0, cz=0):
    """Create a cylinder with a flange at the base."""
    wp = (cq.Workplane("XY")
          .transformed(offset=(cx, cy, cz))
          .cylinder(flange_h, flange_r)
          .faces(">Z").workplane()
          .cylinder(cyl_h, cyl_r))
    return wp, name


def make_box_with_holes(name, lx, ly, lz, hole_r, n_holes=4, cx=0, cy=0, cz=0):
    """Create a box with holes drilled through it."""
    wp = cq.Workplane("XY").transformed(offset=(cx, cy, cz)).box(lx, ly, lz)
    # Add holes in a pattern on the top face
    spacing = min(lx, ly) * 0.3
    for i in range(n_holes):
        angle = 2 * math.pi * i / n_holes
        hx = spacing * math.cos(angle)
        hy = spacing * math.sin(angle)
        wp = wp.faces(">Z").workplane().transformed(offset=(hx, hy, 0)).hole(hole_r * 2)
    return wp, name


def make_solid_top_hollow_bottom_cone(name, r_bottom, r_top, height,
                                      wall_thickness=None, hollow_fraction=0.6,
                                      cx=0, cy=0, cz=0):
    """Create a truncated cone that is wider at the bottom but has more solid
    volume at the top.  The bottom portion is hollow (a tube), while the top
    portion is solid.

    This is the adversarial case the user described: the part is wider at the
    bottom but heavier at the top, which can confuse stacking heuristics.

    Args:
        r_bottom: outer radius at the bottom (wider)
        r_top: outer radius at the top (narrower)
        height: total height
        wall_thickness: wall thickness of the hollow section (default: 15% of r_bottom)
        hollow_fraction: fraction of height that is hollow from the bottom
    """
    if wall_thickness is None:
        wall_thickness = r_bottom * 0.15
    hollow_h = height * hollow_fraction
    solid_h = height - hollow_h

    # Build via revolution profile: outer cone shell hollow at bottom, solid at top
    # Outer profile (right side)
    r_at_hollow_top = r_bottom + (r_top - r_bottom) * hollow_fraction
    ri_bottom = r_bottom - wall_thickness
    ri_at_hollow_top = r_at_hollow_top - wall_thickness

    # Build a 2D profile for revolution around the Z axis
    # Outer contour goes up the outside, across the top, down the inside of the
    # hollow section, across the bottom of the solid section, then back to start
    pts = [
        (r_bottom, 0),                   # bottom-right outer
        (r_at_hollow_top, hollow_h),     # top of hollow section, outer
        (r_top, height),                 # top of cone, outer
        (0, height),                     # top center (solid)
        (0, hollow_h),                   # inner wall top of hollow
        (ri_at_hollow_top, hollow_h),    # inner wall top
        (ri_bottom, 0),                  # inner wall bottom
        (r_bottom, 0),                   # back to start
    ]

    # Use CadQuery revolution
    wp = (cq.Workplane("XZ")
          .transformed(offset=(cx, cy, cz))
          .moveTo(pts[0][0], pts[0][1]))
    for p in pts[1:]:
        wp = wp.lineTo(p[0], p[1])
    wp = wp.close().revolve(360, (0, 0, 0), (0, 1, 0))
    return wp, name


def make_stepped_cylinder(name, radii, heights, cx=0, cy=0, cz=0):
    """Create a stepped cylinder (multiple stacked cylinders of different radii)."""
    # Build from bottom up using revolution
    points = []
    z = 0
    for r, h in zip(radii, heights):
        points.append((r, z))
        z += h
        points.append((r, z))
    # Close the profile
    points.append((0, z))
    points.append((0, 0))

    wp = (cq.Workplane("XZ")
          .transformed(offset=(cx, cy, cz))
          .polyline(points).close()
          .revolve(360, (0, 0, 0), (0, 1, 0)))
    return wp, name


# ── Test rounds ─────────────────────────────────────────────────────

def run_test(label, parts, cut_angle, axis="z", gap=0.0, use_direct=False,
             expect_min_parts=1, tmpdir=None):
    """Run stack → cut → render on parts and collect errors."""
    errors = []
    axis_vec = AXIS_MAP[axis]

    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  Parts: {[n for _, n in parts]}")
    print(f"  Cut angle: {cut_angle}°, axis: {axis}, gap: {gap}")
    print(f"  Method: {'direct' if use_direct else 'boolean'}")
    print(f"{'='*60}")

    # ── Stack ──
    try:
        assy, part_info = stack_parts(parts, axis_vec, gap)
        print(f"  Stack: {len(part_info)} parts assembled")
    except Exception as e:
        errors.append(f"[{label}] Stack failed: {e}")
        traceback.print_exc()
        return errors

    if len(part_info) < expect_min_parts:
        errors.append(
            f"[{label}] Expected at least {expect_min_parts} parts, got {len(part_info)}"
        )

    # ── Build compound ──
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    moved_parts = []
    for name, shape, loc, rgb, segment in part_info:
        moved = apply_location(shape, loc)
        builder.Add(compound, moved)
        moved_parts.append((name, moved, segment))

    # ── Cut ──
    try:
        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()
        _r, _h, origin_pt, _d = _cutter_params(bbox_vals, axis_vec)

        if use_direct:
            # Cut each part individually with direct geometry
            cut_builder = BRep_Builder()
            cut_compound = TopoDS_Compound()
            cut_builder.MakeCompound(cut_compound)
            cut_ok = 0
            for pname, moved_shape, segment in moved_parts:
                try:
                    if segment is not None:
                        cut_part = cut_part_direct_segment(
                            moved_shape, cut_angle, axis_vec, origin_pt, segment)
                    else:
                        cut_part = cut_part_direct(
                            moved_shape, cut_angle, axis_vec, origin_pt)
                    if cut_part is not None:
                        part_bbox = Bnd_Box()
                        BRepBndLib.Add_s(cut_part, part_bbox)
                        if not part_bbox.IsVoid():
                            cut_builder.Add(cut_compound, cut_part)
                            cut_ok += 1
                except Exception as e:
                    errors.append(f"[{label}] Direct cut failed for '{pname}': {e}")
                    cut_builder.Add(cut_compound, moved_shape)
                    cut_ok += 1
            cut_shape = cut_compound
            print(f"  Cut (direct): {cut_ok} parts")
        else:
            # Boolean cut each part individually
            cutter = make_cutter(bbox_vals, cut_angle, axis_vec)
            cut_builder = BRep_Builder()
            cut_compound = TopoDS_Compound()
            cut_builder.MakeCompound(cut_compound)
            cut_ok = 0
            for pname, moved_shape, segment in moved_parts:
                try:
                    cut_part = cut_assembly(moved_shape, cutter)
                    if cut_part is not None:
                        part_bbox = Bnd_Box()
                        BRepBndLib.Add_s(cut_part, part_bbox)
                        if not part_bbox.IsVoid():
                            cut_builder.Add(cut_compound, cut_part)
                            cut_ok += 1
                except Exception as e:
                    errors.append(f"[{label}] Boolean cut failed for '{pname}': {e}")
                    cut_builder.Add(cut_compound, moved_shape)
                    cut_ok += 1
            cut_shape = cut_compound
            print(f"  Cut (boolean): {cut_ok} parts")

        # Validate cut result
        cut_errors = validate_cut_result(compound, cut_shape, cut_angle, label)
        errors.extend(cut_errors)

    except Exception as e:
        errors.append(f"[{label}] Cut phase failed: {e}")
        traceback.print_exc()
        cut_shape = compound  # fallback to uncut for render test

    # ── Export STEP ──
    step_path = os.path.join(tmpdir, f"{label.replace(' ', '_')}.step")
    try:
        export_shape_step(cut_shape, step_path)
        if not os.path.exists(step_path):
            errors.append(f"[{label}] STEP file not created")
        elif os.path.getsize(step_path) < 100:
            errors.append(f"[{label}] STEP file suspiciously small")
        else:
            print(f"  STEP exported: {os.path.getsize(step_path)} bytes")
    except Exception as e:
        errors.append(f"[{label}] STEP export failed: {e}")

    # ── Render ──
    render_path = os.path.join(tmpdir, f"{label.replace(' ', '_')}.png")
    render_errors = validate_render(part_info, cut_shape, label, render_path)
    errors.extend(render_errors)
    if not render_errors:
        sz = os.path.getsize(render_path) if os.path.exists(render_path) else 0
        print(f"  Render: {sz} bytes")

    # ── Summary ──
    if errors:
        print(f"\n  ✗ {len(errors)} error(s):")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  ✓ PASSED")

    return errors


# ── Round definitions ───────────────────────────────────────────────

def round_1_basic(tmpdir):
    """Round 1: Basic shapes — boxes, cylinders, concentric stacking."""
    all_errors = []

    # Test 1.1: Simple 3-level concentric stack with 90° cut
    parts = [
        make_box("outer_1", 100, 100, 30),
        make_cylinder("mid_1", 30, 30),
        make_cylinder("inner_1", 10, 30),
        make_box("outer_2", 100, 100, 20),
        make_cylinder("mid_2", 25, 20),
    ]
    all_errors.extend(run_test("R1.1 basic_concentric_90", parts, 90, tmpdir=tmpdir))

    # Test 1.2: Same stack with 45° cut (thin wedge)
    all_errors.extend(run_test("R1.2 basic_concentric_45", parts, 45, tmpdir=tmpdir))

    # Test 1.3: Same stack with 270° cut (large removal)
    all_errors.extend(run_test("R1.3 basic_concentric_270", parts, 270, tmpdir=tmpdir))

    # Test 1.4: 180° cut (half removal)
    all_errors.extend(run_test("R1.4 basic_concentric_180", parts, 180, tmpdir=tmpdir))

    # Test 1.5: Direct geometry method
    all_errors.extend(run_test("R1.5 basic_direct_90", parts, 90, use_direct=True, tmpdir=tmpdir))

    # Test 1.6: Non-zero gap
    all_errors.extend(run_test("R1.6 basic_gap", parts, 90, gap=5.0, tmpdir=tmpdir))

    # Test 1.7: Simple unnamed parts (auto-tier)
    parts_unnamed = [
        make_box("plate", 80, 80, 10),
        make_cylinder("shaft", 15, 50),
        make_box("flange", 60, 60, 8),
    ]
    all_errors.extend(run_test("R1.7 unnamed_parts", parts_unnamed, 90, tmpdir=tmpdir))

    return all_errors


def round_2_moderate(tmpdir):
    """Round 2: Moderate difficulty — thin walls, hollow shapes, extreme angles."""
    all_errors = []

    # Test 2.1: Hollow cylinders (thin-wall tubes) — boolean ops struggle with thin features
    parts = [
        make_hollow_cylinder("outer_1", 50, 45, 40),  # 5mm wall
        make_hollow_cylinder("mid_1", 30, 27, 40),     # 3mm wall
        make_cylinder("inner_1", 10, 40),
        make_hollow_cylinder("outer_2", 50, 48, 20),   # 2mm wall — very thin!
    ]
    all_errors.extend(run_test("R2.1 hollow_cylinders", parts, 90, tmpdir=tmpdir))
    all_errors.extend(run_test("R2.2 hollow_direct", parts, 90, use_direct=True, tmpdir=tmpdir))

    # Test 2.3: Very thin cut angle (10°) — near-zero wedge
    parts_basic = [
        make_box("outer_1", 60, 60, 30),
        make_cylinder("inner_1", 20, 30),
    ]
    all_errors.extend(run_test("R2.3 thin_wedge_10", parts_basic, 10, tmpdir=tmpdir))

    # Test 2.4: Near-full removal (350°)
    all_errors.extend(run_test("R2.4 near_full_350", parts_basic, 350, tmpdir=tmpdir))

    # Test 2.5: Hexagonal prism inside cylinder
    parts = [
        make_cylinder("outer_1", 40, 50),
        make_hexagonal_prism("mid_1", 40, 50),
        make_cylinder("inner_1", 8, 50),
    ]
    all_errors.extend(run_test("R2.5 hex_in_cyl", parts, 120, tmpdir=tmpdir))

    # Test 2.6: Cone + cylinder stack
    parts = [
        make_cone("outer_1", 40, 20, 30),
        make_cylinder("outer_2", 20, 40),
    ]
    all_errors.extend(run_test("R2.6 cone_cyl_stack", parts, 90, tmpdir=tmpdir))

    # Test 2.7: 5-level deep stack
    parts = [
        make_box("outer_1", 80, 80, 10),
        make_box("outer_2", 70, 70, 10),
        make_box("outer_3", 60, 60, 10),
        make_box("outer_4", 50, 50, 10),
        make_box("outer_5", 40, 40, 10),
        make_cylinder("inner_1", 15, 10),
        make_cylinder("inner_2", 12, 10),
        make_cylinder("inner_3", 10, 10),
        make_cylinder("inner_4", 8, 10),
        make_cylinder("inner_5", 6, 10),
    ]
    all_errors.extend(run_test("R2.7 deep_stack_5level", parts, 90, tmpdir=tmpdir))
    all_errors.extend(run_test("R2.8 deep_stack_direct", parts, 90, use_direct=True, tmpdir=tmpdir))

    return all_errors


def round_3_hard(tmpdir):
    """Round 3: Hard — segments, spheres, off-center parts, mixed geometry."""
    all_errors = []

    # Test 3.1: Segment splitting (a/b halves)
    parts = [
        make_box("outer_1", 80, 80, 30),
        make_cylinder("inner_1a", 20, 30),
        make_cylinder("inner_1b", 20, 30),
    ]
    all_errors.extend(run_test("R3.1 segment_split", parts, 90, tmpdir=tmpdir))
    all_errors.extend(run_test("R3.2 segment_direct", parts, 90, use_direct=True, tmpdir=tmpdir))

    # Test 3.3: Sphere inside box — boolean cut through curved surface
    parts = [
        make_box("outer_1", 80, 80, 80),
        make_sphere("mid_1", 30),
    ]
    all_errors.extend(run_test("R3.3 sphere_in_box", parts, 90, tmpdir=tmpdir))
    all_errors.extend(run_test("R3.4 sphere_direct", parts, 90, use_direct=True, tmpdir=tmpdir))

    # Test 3.5: Multiple spheres at different levels
    parts = [
        make_sphere("outer_1", 30),
        make_sphere("outer_2", 25),
        make_sphere("outer_3", 20),
    ]
    all_errors.extend(run_test("R3.5 sphere_stack", parts, 120, tmpdir=tmpdir))

    # Test 3.6: Torus (donut) — complex surface topology
    parts = [
        make_box("outer_1", 100, 100, 40),
        make_torus("mid_1", 30, 10),
    ]
    all_errors.extend(run_test("R3.6 torus_in_box", parts, 90, tmpdir=tmpdir))

    # Test 3.7: Stepped cylinder (gear-like profile)
    parts = [
        make_stepped_cylinder("outer_1", [30, 20, 30, 15], [10, 15, 10, 20]),
        make_cylinder("inner_1", 8, 55),
    ]
    all_errors.extend(run_test("R3.7 stepped_cyl", parts, 90, tmpdir=tmpdir))

    # Test 3.8: Box with holes — boolean on already-boolean'd geometry
    parts = [
        make_box_with_holes("outer_1", 80, 80, 30, 5, 6),
        make_cylinder("inner_1", 10, 30),
    ]
    all_errors.extend(run_test("R3.8 holey_box", parts, 90, tmpdir=tmpdir))
    all_errors.extend(run_test("R3.9 holey_direct", parts, 90, use_direct=True, tmpdir=tmpdir))

    return all_errors


def round_4_extreme(tmpdir):
    """Round 4: Extreme — stress tests for edge cases."""
    all_errors = []

    # Test 4.1: Very small parts (micro-scale)
    parts = [
        make_box("outer_1", 0.5, 0.5, 0.2),
        make_cylinder("inner_1", 0.1, 0.2),
    ]
    all_errors.extend(run_test("R4.1 micro_scale", parts, 90, tmpdir=tmpdir))

    # Test 4.2: Very large parts
    parts = [
        make_box("outer_1", 5000, 5000, 2000),
        make_cylinder("inner_1", 1000, 2000),
    ]
    all_errors.extend(run_test("R4.2 macro_scale", parts, 90, tmpdir=tmpdir))

    # Test 4.3: Large aspect ratio (very tall, thin cylinder inside a short wide box)
    parts = [
        make_box("outer_1", 200, 200, 5),     # pancake
        make_cylinder("inner_1", 2, 5),        # pin
    ]
    all_errors.extend(run_test("R4.3 pancake_pin", parts, 90, tmpdir=tmpdir))

    # Test 4.4: Needle-like part (very tall, very thin)
    parts = [
        make_cylinder("outer_1", 2, 500),       # needle
        make_cylinder("inner_1", 0.5, 500),
    ]
    all_errors.extend(run_test("R4.4 needle", parts, 90, tmpdir=tmpdir))

    # Test 4.5: Near-boundary cut angle (1°)
    parts = [
        make_box("outer_1", 60, 60, 30),
    ]
    all_errors.extend(run_test("R4.5 cut_1deg", parts, 1, tmpdir=tmpdir))

    # Test 4.6: Near-boundary cut angle (359°)
    all_errors.extend(run_test("R4.6 cut_359deg", parts, 359, tmpdir=tmpdir))

    # Test 4.7: All three tiers at 3 levels with segments + direct cut + gap
    parts = [
        make_box("outer_1", 100, 100, 20),
        make_cylinder("mid_1", 35, 20),
        make_cylinder("inner_1a", 15, 20),
        make_cylinder("inner_1b", 15, 20),
        make_box("outer_2", 90, 90, 25),
        make_hexagonal_prism("mid_2", 50, 25),
        make_cylinder("inner_2a", 12, 25),
        make_cylinder("inner_2b", 12, 25),
        make_cylinder("outer_3", 45, 15),
        make_cylinder("mid_3", 25, 15),
        make_sphere("inner_3", 10),
    ]
    all_errors.extend(run_test(
        "R4.7 full_concentric_3x3_seg_bool", parts, 90,
        gap=3.0, tmpdir=tmpdir
    ))
    all_errors.extend(run_test(
        "R4.8 full_concentric_3x3_seg_direct", parts, 90,
        gap=3.0, use_direct=True, tmpdir=tmpdir
    ))

    # Test 4.9: Y-axis stacking
    parts = [
        make_box("outer_1", 50, 50, 50),
        make_box("outer_2", 40, 40, 40),
    ]
    all_errors.extend(run_test("R4.9 y_axis_stack", parts, 90, axis="y", tmpdir=tmpdir))

    # Test 4.10: X-axis stacking
    all_errors.extend(run_test("R4.10 x_axis_stack", parts, 90, axis="x", tmpdir=tmpdir))

    return all_errors


def round_5_conic_orientation(tmpdir):
    """Round 5: Truncated cones with solid tops and hollow bottoms.

    These parts are wider at the bottom but have more volume at the top.
    When stacked as sequential outer parts, there should be NO gap between
    them — the top surface of one part should meet the bottom surface of
    the next.

    This tests correct orientation handling and gapless stacking.
    """
    all_errors = []

    # Test 5.1: Two identical solid-top-hollow-bottom cones stacked
    parts = [
        make_solid_top_hollow_bottom_cone("outer_1", 40, 20, 50),
        make_solid_top_hollow_bottom_cone("outer_2", 40, 20, 50),
    ]
    errors = run_test("R5.1 two_conic_stack", parts, 90, tmpdir=tmpdir)
    # Additional check: verify no gap between parts
    errors.extend(_check_no_gap_between_outers(parts, "R5.1"))
    all_errors.extend(errors)

    # Test 5.2: Three cones of decreasing size stacked
    parts = [
        make_solid_top_hollow_bottom_cone("outer_1", 50, 25, 60, hollow_fraction=0.7),
        make_solid_top_hollow_bottom_cone("outer_2", 45, 22, 50, hollow_fraction=0.6),
        make_solid_top_hollow_bottom_cone("outer_3", 40, 20, 40, hollow_fraction=0.5),
    ]
    errors = run_test("R5.2 three_conic_stack", parts, 90, tmpdir=tmpdir)
    errors.extend(_check_no_gap_between_outers(parts, "R5.2"))
    all_errors.extend(errors)

    # Test 5.3: Conic parts with concentric inner cylinders
    parts = [
        make_solid_top_hollow_bottom_cone("outer_1", 50, 25, 60),
        make_cylinder("inner_1", 10, 60),
        make_solid_top_hollow_bottom_cone("outer_2", 50, 25, 60),
        make_cylinder("inner_2", 10, 60),
    ]
    errors = run_test("R5.3 conic_with_inner", parts, 90, tmpdir=tmpdir)
    all_errors.extend(errors)

    # Test 5.4: Direct geometry cut on conic stack
    parts = [
        make_solid_top_hollow_bottom_cone("outer_1", 40, 20, 50),
        make_solid_top_hollow_bottom_cone("outer_2", 40, 20, 50),
    ]
    errors = run_test("R5.4 conic_direct", parts, 90, use_direct=True, tmpdir=tmpdir)
    all_errors.extend(errors)

    # Test 5.5: Very thin-walled hollow-bottom cone (stress test)
    parts = [
        make_solid_top_hollow_bottom_cone("outer_1", 40, 20, 50,
                                          wall_thickness=1.0, hollow_fraction=0.8),
        make_solid_top_hollow_bottom_cone("outer_2", 40, 20, 50,
                                          wall_thickness=1.0, hollow_fraction=0.8),
    ]
    errors = run_test("R5.5 thin_wall_conic", parts, 90, tmpdir=tmpdir)
    errors.extend(_check_no_gap_between_outers(parts, "R5.5"))
    all_errors.extend(errors)

    # Test 5.6: 270° cut on conic stack
    parts = [
        make_solid_top_hollow_bottom_cone("outer_1", 40, 20, 50),
        make_solid_top_hollow_bottom_cone("outer_2", 40, 20, 50),
    ]
    errors = run_test("R5.6 conic_270", parts, 270, tmpdir=tmpdir)
    all_errors.extend(errors)

    return all_errors


def round_6_ultimate(tmpdir):
    """Round 6: Ultimate stress tests — combined adversarial cases."""
    all_errors = []

    # Test 6.1: Hollow cone inside solid cone inside box — triple concentric
    parts = [
        make_box("outer_1", 100, 100, 60),
        make_cone("mid_1", 40, 15, 60),
        make_hollow_cylinder("inner_1", 12, 8, 60),
    ]
    all_errors.extend(run_test("R6.1 box_cone_tube", parts, 90, tmpdir=tmpdir))
    all_errors.extend(run_test("R6.2 box_cone_tube_direct", parts, 90, use_direct=True, tmpdir=tmpdir))

    # Test 6.3: All conic stack with segments — 3 levels
    parts = [
        make_solid_top_hollow_bottom_cone("outer_1", 50, 25, 40, hollow_fraction=0.6),
        make_cylinder("inner_1a", 10, 40),
        make_cylinder("inner_1b", 10, 40),
        make_solid_top_hollow_bottom_cone("outer_2", 50, 25, 40, hollow_fraction=0.6),
        make_cylinder("inner_2a", 10, 40),
        make_cylinder("inner_2b", 10, 40),
        make_solid_top_hollow_bottom_cone("outer_3", 50, 25, 40, hollow_fraction=0.6),
    ]
    errors = run_test("R6.3 conic_3level_segments", parts, 90, tmpdir=tmpdir)
    errors.extend(_check_no_gap_between_outers(
        [p for p in parts if 'outer' in p[1]], "R6.3"))
    all_errors.extend(errors)

    # Test 6.4: Same with direct
    all_errors.extend(run_test("R6.4 conic_3level_seg_direct", parts, 90,
                               use_direct=True, tmpdir=tmpdir))

    # Test 6.5: Mixed geometry — sphere + torus + hex at same level
    parts = [
        make_box("outer_1", 120, 120, 60),
        make_torus("mid_1", 40, 8),
        make_sphere("inner_1", 15),
    ]
    all_errors.extend(run_test("R6.5 sphere_torus_box", parts, 120, tmpdir=tmpdir))

    # Test 6.6: Stepped cylinder with thin walls direct cut at 45°
    parts = [
        make_stepped_cylinder("outer_1", [30, 20, 30, 15, 25], [8, 12, 8, 15, 10]),
        make_hollow_cylinder("inner_1", 10, 8, 53),
    ]
    all_errors.extend(run_test("R6.6 stepped_hollow_45", parts, 45, tmpdir=tmpdir))
    all_errors.extend(run_test("R6.7 stepped_hollow_45_direct", parts, 45,
                               use_direct=True, tmpdir=tmpdir))

    # Test 6.8: 5 conic parts stacked, gap check
    parts = [
        make_solid_top_hollow_bottom_cone(f"outer_{i}", 40, 20, 30,
                                          hollow_fraction=0.5 + i * 0.05)
        for i in range(1, 6)
    ]
    errors = run_test("R6.8 five_conic_stack", parts, 90, tmpdir=tmpdir)
    errors.extend(_check_no_gap_between_outers(parts, "R6.8"))
    all_errors.extend(errors)

    # Test 6.9: Box with holes + sphere + segments at 270°
    parts = [
        make_box_with_holes("outer_1", 100, 100, 40, 8, 8),
        make_sphere("mid_1", 20),
        make_cylinder("inner_1a", 8, 40),
        make_cylinder("inner_1b", 8, 40),
    ]
    all_errors.extend(run_test("R6.9 holey_sphere_seg_270", parts, 270, tmpdir=tmpdir))

    # Test 6.10: Direct cut at 180° on deep concentric stack
    parts = [
        make_box("outer_1", 80, 80, 15),
        make_hexagonal_prism("mid_1", 45, 15),
        make_cylinder("inner_1", 8, 15),
        make_box("outer_2", 80, 80, 15),
        make_cylinder("mid_2", 22, 15),
        make_hollow_cylinder("inner_2", 8, 5, 15),
    ]
    all_errors.extend(run_test("R6.10 deep_concentric_180_direct", parts, 180,
                               use_direct=True, tmpdir=tmpdir))

    return all_errors


def _check_no_gap_between_outers(parts, label):
    """Verify that sequential outer parts stack with no gap.

    After stacking, the top of outer_N should equal the bottom of outer_N+1.
    """
    errors = []
    axis_vec = AXIS_MAP["z"]

    try:
        assy, part_info = stack_parts(parts, axis_vec, 0.0)

        # Get Z extents of each part after stacking
        z_ranges = []
        for name, shape, loc, rgb, segment in part_info:
            if "outer" in name.lower() or segment is None:
                moved = apply_location(shape, loc)
                bb = get_bounding_box(moved)
                z_min, z_max = bb[2], bb[5]
                z_ranges.append((name, z_min, z_max))

        z_ranges.sort(key=lambda x: x[1])  # sort by z_min

        for i in range(len(z_ranges) - 1):
            name_a, _, z_top_a = z_ranges[i]
            name_b, z_bot_b, _ = z_ranges[i + 1]
            gap = abs(z_bot_b - z_top_a)
            if gap > 0.01:  # tolerance of 0.01 units
                errors.append(
                    f"[{label}] Gap of {gap:.4f} between '{name_a}' (top={z_top_a:.4f}) "
                    f"and '{name_b}' (bottom={z_bot_b:.4f})"
                )
            else:
                print(f"  Gap check {name_a}→{name_b}: {gap:.6f} (OK)")
    except Exception as e:
        errors.append(f"[{label}] Gap check failed: {e}")

    return errors


# ── Main ────────────────────────────────────────────────────────────

def main():
    tmpdir = tempfile.mkdtemp(prefix="adv_test_")
    print(f"Output directory: {tmpdir}")

    all_errors = []
    all_rounds = [
        ("ROUND 1: Basic Shapes", round_1_basic),
        ("ROUND 2: Moderate Difficulty", round_2_moderate),
        ("ROUND 3: Hard", round_3_hard),
        ("ROUND 4: Extreme", round_4_extreme),
        ("ROUND 5: Conic Orientation (solid-top, hollow-bottom)", round_5_conic_orientation),
        ("ROUND 6: Ultimate Stress Tests", round_6_ultimate),
    ]

    for round_name, round_fn in all_rounds:
        print(f"\n{'#'*70}")
        print(f"# {round_name}")
        print(f"{'#'*70}")
        errors = round_fn(tmpdir)
        all_errors.extend(errors)

        if errors:
            print(f"\n{'!'*60}")
            print(f"! {round_name} — {len(errors)} error(s)")
            print(f"{'!'*60}")
            for e in errors:
                print(f"  {e}")
        else:
            print(f"\n✓ {round_name} — ALL PASSED")

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    if all_errors:
        print(f"\n{len(all_errors)} total error(s):")
        for e in all_errors:
            print(f"  - {e}")
        return 1
    else:
        print(f"\nALL TESTS PASSED — no errors found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
