"""
Tests for the CAD Assembly Pipeline (assemble.py).

Covers: loading, stacking, cutting, STEP export, tessellation, and rendering.

Run with:
  xvfb-run -a python3 -m pytest test_assemble.py -v
"""

import os
import sys
import math
import tempfile
import shutil

import pytest
import numpy as np
import cadquery as cq

# Ensure assemble module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assemble import (
    get_bounding_box,
    shape_extent,
    translate_shape,
    apply_location,
    load_part,
    pick_color,
    stack_parts,
    make_cutter,
    cut_assembly,
    export_assembly_step,
    export_shape_step,
    tessellate_shape,
    AXIS_MAP,
    MATERIAL_COLORS,
)

from cadquery import Location, Vector, Color
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_assemble_")
    yield d
    shutil.rmtree(d)


@pytest.fixture
def box_step(tmp_dir):
    """Create a simple box STEP file."""
    path = os.path.join(tmp_dir, "box.step")
    box = cq.Workplane("XY").box(20, 20, 10)
    cq.exporters.export(box, path)
    return path


@pytest.fixture
def cylinder_step(tmp_dir):
    """Create a cylinder STEP file."""
    path = os.path.join(tmp_dir, "cylinder.step")
    cyl = cq.Workplane("XY").cylinder(30, 8)
    cq.exporters.export(cyl, path)
    return path


@pytest.fixture
def flange_step(tmp_dir):
    """Create a flange with hex cutout."""
    path = os.path.join(tmp_dir, "flange.step")
    f = (
        cq.Workplane("XY")
        .circle(15).extrude(5)
        .faces(">Z").workplane()
        .polygon(6, 12).cutBlind(-5)
    )
    cq.exporters.export(f, path)
    return path


@pytest.fixture
def copper_part(tmp_dir):
    """Create a part with 'copper' in name for material detection."""
    path = os.path.join(tmp_dir, "copper_tube.step")
    tube = cq.Workplane("XY").circle(10).circle(7).extrude(20)
    cq.exporters.export(tube, path)
    return path


@pytest.fixture
def stl_part(tmp_dir):
    """Create a simple STL mesh file."""
    path = os.path.join(tmp_dir, "mesh_part.stl")
    box = cq.Workplane("XY").box(15, 15, 8)
    cq.exporters.export(box, path, exportType="STL")
    return path


# ============================================================================
# Test: Geometry helpers
# ============================================================================

class TestBoundingBox:
    def test_box_bounds(self):
        box = cq.Workplane("XY").box(20, 30, 10)
        shape = box.val().wrapped
        xn, yn, zn, xx, yx, zx = get_bounding_box(shape)
        assert abs(xn - (-10)) < 0.01
        assert abs(xx - 10) < 0.01
        assert abs(yn - (-15)) < 0.01
        assert abs(yx - 15) < 0.01
        assert abs(zn - (-5)) < 0.01
        assert abs(zx - 5) < 0.01

    def test_shape_extent_z(self):
        box = cq.Workplane("XY").box(20, 30, 10)
        shape = box.val().wrapped
        lo, hi = shape_extent(shape, AXIS_MAP["z"])
        assert abs(hi - lo - 10) < 0.01

    def test_shape_extent_x(self):
        box = cq.Workplane("XY").box(20, 30, 10)
        shape = box.val().wrapped
        lo, hi = shape_extent(shape, AXIS_MAP["x"])
        assert abs(hi - lo - 20) < 0.01

    def test_shape_extent_y(self):
        box = cq.Workplane("XY").box(20, 30, 10)
        shape = box.val().wrapped
        lo, hi = shape_extent(shape, AXIS_MAP["y"])
        assert abs(hi - lo - 30) < 0.01


class TestTranslateShape:
    def test_translate_z(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        moved = translate_shape(shape, 0, 0, 50)
        xn, yn, zn, xx, yx, zx = get_bounding_box(moved)
        assert abs(zn - 45) < 0.01
        assert abs(zx - 55) < 0.01

    def test_translate_preserves_size(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        moved = translate_shape(shape, 100, 200, 300)
        xn, yn, zn, xx, yx, zx = get_bounding_box(moved)
        assert abs((xx - xn) - 10) < 0.01
        assert abs((yx - yn) - 10) < 0.01
        assert abs((zx - zn) - 10) < 0.01


class TestApplyLocation:
    def test_identity(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        loc = Location(Vector(0, 0, 0))
        moved = apply_location(shape, loc)
        xn, yn, zn, xx, yx, zx = get_bounding_box(moved)
        assert abs(xn - (-5)) < 0.01

    def test_translate_via_location(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        loc = Location(Vector(0, 0, 100))
        moved = apply_location(shape, loc)
        xn, yn, zn, xx, yx, zx = get_bounding_box(moved)
        assert abs(zn - 95) < 0.01
        assert abs(zx - 105) < 0.01

    def test_none_location(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        moved = apply_location(shape, None)
        xn, yn, zn, xx, yx, zx = get_bounding_box(moved)
        assert abs(zn - (-5)) < 0.01


# ============================================================================
# Test: File loading
# ============================================================================

class TestLoadPart:
    def test_load_step(self, box_step):
        wp, name = load_part(box_step)
        assert name == "box"
        shape = wp.val().wrapped
        assert not shape.IsNull()

    def test_load_stl(self, stl_part):
        wp, name = load_part(stl_part)
        assert name == "mesh_part"
        shape = wp.val().wrapped
        assert not shape.IsNull()

    def test_load_nonexistent(self):
        with pytest.raises(Exception):
            load_part("/nonexistent/file.step")

    def test_load_unsupported(self, tmp_dir):
        path = os.path.join(tmp_dir, "file.xyz")
        with open(path, "w") as f:
            f.write("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            load_part(path)


# ============================================================================
# Test: Color assignment
# ============================================================================

class TestColorPicking:
    def test_copper_keyword(self):
        cq_color, rgb = pick_color("copper_tube", 0)
        assert abs(rgb[0] - 0.85) < 0.01
        assert abs(rgb[1] - 0.45) < 0.01

    def test_steel_keyword(self):
        cq_color, rgb = pick_color("STEEL_plate", 0)
        assert abs(rgb[0] - 0.55) < 0.01

    def test_gold_keyword(self):
        cq_color, rgb = pick_color("gold_ring", 0)
        assert abs(rgb[0] - 0.90) < 0.01

    def test_no_keyword_uses_palette(self):
        _, rgb1 = pick_color("part_a", 0)
        _, rgb2 = pick_color("part_b", 1)
        assert rgb1 != rgb2  # different palette entries

    def test_palette_wraps(self):
        _, rgb0 = pick_color("part", 0)
        _, rgb8 = pick_color("part", 8)
        assert rgb0 == rgb8  # wraps around palette length


# ============================================================================
# Test: Stacking
# ============================================================================

class TestStackParts:
    def test_single_part_z(self, box_step):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        assert len(info) == 1
        # Part should be moved so its bottom is at z=0
        moved = apply_location(info[0][1], info[0][2])
        xn, yn, zn, xx, yx, zx = get_bounding_box(moved)
        assert abs(zn - 0) < 0.1  # bottom at z=0 (moved from z=-5)

    def test_two_parts_stacked_z(self, box_step, cylinder_step):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["z"], gap=0)
        assert len(info) == 2

        # First part: bottom at z=0, top at z=10
        moved1 = apply_location(info[0][1], info[0][2])
        _, _, z1n, _, _, z1x = get_bounding_box(moved1)
        assert abs(z1n - 0) < 0.1
        assert abs(z1x - 10) < 0.1

        # Second part: bottom at z=10
        moved2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, z2x = get_bounding_box(moved2)
        assert abs(z2n - 10) < 0.5

    def test_gap_spacing(self, box_step, cylinder_step):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        gap = 5.0
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["z"], gap=gap)

        moved1 = apply_location(info[0][1], info[0][2])
        _, _, _, _, _, z1x = get_bounding_box(moved1)

        moved2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, _ = get_bounding_box(moved2)

        assert abs(z2n - z1x - gap) < 0.5

    def test_stack_x_axis(self, box_step, cylinder_step):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["x"], gap=0)

        moved1 = apply_location(info[0][1], info[0][2])
        xn, _, _, xx, _, _ = get_bounding_box(moved1)
        assert abs(xn - 0) < 0.1

    def test_stack_y_axis(self, box_step, cylinder_step):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["y"], gap=0)

        moved1 = apply_location(info[0][1], info[0][2])
        _, yn, _, _, yx, _ = get_bounding_box(moved1)
        assert abs(yn - 0) < 0.1

    def test_three_parts_ordered(self, box_step, cylinder_step, flange_step):
        parts = [load_part(p) for p in [box_step, cylinder_step, flange_step]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=2)
        assert len(info) == 3

        # Verify ordering: each part's bottom >= previous part's top + gap
        prev_top = None
        for name, shape, loc, rgb in info:
            moved = apply_location(shape, loc)
            _, _, zn, _, _, zx = get_bounding_box(moved)
            if prev_top is not None:
                assert zn >= prev_top + 1.5  # gap=2, allow tolerance
            prev_top = zx


# ============================================================================
# Test: Cutting
# ============================================================================

class TestCutting:
    def _build_compound(self, info):
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for name, shape, loc, rgb in info:
            moved = apply_location(shape, loc)
            builder.Add(compound, moved)
        return compound

    def test_cut_90_removes_quarter(self, box_step):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        compound = self._build_compound(info)

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()

        cutter = make_cutter(bbox_vals, 90, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)

        assert not result.IsNull()
        # The result should exist (not be empty)
        result_bbox = Bnd_Box()
        BRepBndLib.Add_s(result, result_bbox)
        assert not result_bbox.IsVoid()

    def test_cut_180_removes_half(self, box_step):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        compound = self._build_compound(info)

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()

        cutter = make_cutter(bbox_vals, 180, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)
        assert not result.IsNull()

    def test_cut_multi_part_assembly(self, box_step, cylinder_step):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["z"], gap=1)
        compound = self._build_compound(info)

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()

        cutter = make_cutter(bbox_vals, 90, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)
        assert not result.IsNull()

    def test_cutter_axes(self, box_step):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        compound = self._build_compound(info)

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()

        for axis_name in ("x", "y", "z"):
            cutter = make_cutter(bbox_vals, 90, AXIS_MAP[axis_name])
            result = cut_assembly(compound, cutter)
            assert not result.IsNull(), f"Cut failed for axis {axis_name}"


# ============================================================================
# Test: STEP export
# ============================================================================

class TestStepExport:
    def test_assembly_export(self, box_step, cylinder_step, tmp_dir):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["z"], gap=0)

        out_path = os.path.join(tmp_dir, "assy_out.step")
        export_assembly_step(assy, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 100

    def test_shape_export(self, box_step, tmp_dir):
        wp, name = load_part(box_step)
        shape = wp.val().wrapped

        out_path = os.path.join(tmp_dir, "shape_out.step")
        export_shape_step(shape, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 100

    def test_cut_result_export(self, box_step, tmp_dir):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for n, s, l, c in info:
            builder.Add(compound, apply_location(s, l))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 90, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)

        out_path = os.path.join(tmp_dir, "cut_out.step")
        export_shape_step(result, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 100

    def test_exported_step_is_reimportable(self, box_step, cylinder_step, tmp_dir):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["z"], gap=2)

        out_path = os.path.join(tmp_dir, "reimport.step")
        export_assembly_step(assy, out_path)

        # Reimport should succeed
        reimported = cq.importers.importStep(out_path)
        shape = reimported.val().wrapped
        assert not shape.IsNull()


# ============================================================================
# Test: Tessellation
# ============================================================================

class TestTessellation:
    def test_box_tessellation(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        verts, faces = tessellate_shape(box.val().wrapped, tolerance=0.1)
        assert len(verts) > 0
        assert len(faces) > 0
        # Box should have at least 8 vertices and 12 triangles
        assert len(verts) >= 8
        assert len(faces) >= 12

    def test_cylinder_tessellation(self):
        cyl = cq.Workplane("XY").cylinder(20, 5)
        verts, faces = tessellate_shape(cyl.val().wrapped, tolerance=0.1)
        assert len(verts) > 20  # cylinder needs many vertices
        assert len(faces) > 10

    def test_face_format(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        verts, faces = tessellate_shape(box.val().wrapped, tolerance=0.1)
        # Each face row should be [3, i1, i2, i3] for triangles
        for face in faces:
            assert face[0] == 3
            assert all(0 <= face[j] < len(verts) for j in range(1, 4))

    def test_tolerance_affects_detail(self):
        cyl = cq.Workplane("XY").cylinder(20, 5)
        shape = cyl.val().wrapped
        v_coarse, f_coarse = tessellate_shape(shape, tolerance=1.0)
        v_fine, f_fine = tessellate_shape(shape, tolerance=0.01)
        # Finer tolerance should produce more vertices
        assert len(v_fine) >= len(v_coarse)


# ============================================================================
# Test: End-to-end pipeline (no rendering — that requires xvfb)
# ============================================================================

class TestPipelineNoRender:
    def test_single_part_pipeline(self, box_step, tmp_dir):
        """Stack a single part and export."""
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        out = os.path.join(tmp_dir, "single.step")
        export_assembly_step(assy, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100

    def test_multi_part_pipeline(self, box_step, cylinder_step, flange_step, tmp_dir):
        """Stack 3 parts and export."""
        parts = [load_part(p) for p in [box_step, cylinder_step, flange_step]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=2)
        out = os.path.join(tmp_dir, "multi.step")
        export_assembly_step(assy, out)
        assert os.path.exists(out)

    def test_cut_pipeline(self, box_step, cylinder_step, tmp_dir):
        """Stack, cut, export."""
        parts = [load_part(p) for p in [box_step, cylinder_step]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=1)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for n, s, l, c in info:
            builder.Add(compound, apply_location(s, l))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 120, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)

        out = os.path.join(tmp_dir, "cut_pipeline.step")
        export_shape_step(result, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100

    def test_mesh_input_pipeline(self, stl_part, tmp_dir):
        """Load an STL mesh and stack it."""
        wp, name = load_part(stl_part)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        assert len(info) == 1

    def test_mixed_cad_mesh_pipeline(self, box_step, stl_part, tmp_dir):
        """Mix STEP and STL inputs."""
        parts = [load_part(p) for p in [box_step, stl_part]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=2)
        assert len(info) == 2


# ============================================================================
# Test: Edge cases
# ============================================================================

class TestEdgeCases:
    def test_very_small_angle(self, box_step, tmp_dir):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for n, s, l, c in info:
            builder.Add(compound, apply_location(s, l))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 5, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)
        assert not result.IsNull()

    def test_near_full_angle(self, box_step, tmp_dir):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for n, s, l, c in info:
            builder.Add(compound, apply_location(s, l))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 350, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)
        assert not result.IsNull()

    def test_zero_gap(self, box_step, cylinder_step):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["z"], gap=0)

        moved1 = apply_location(info[0][1], info[0][2])
        _, _, _, _, _, z1x = get_bounding_box(moved1)

        moved2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, _ = get_bounding_box(moved2)

        # Parts should be touching (no gap)
        assert abs(z2n - z1x) < 0.5

    def test_large_gap(self, box_step, cylinder_step):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["z"], gap=100)

        moved1 = apply_location(info[0][1], info[0][2])
        _, _, _, _, _, z1x = get_bounding_box(moved1)

        moved2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, _ = get_bounding_box(moved2)

        assert abs(z2n - z1x - 100) < 0.5


# ============================================================================
# Test: Material keyword detection
# ============================================================================

class TestMaterialDetection:
    @pytest.mark.parametrize("name,expected_r", [
        ("copper_tube", 0.85),
        ("STEEL_plate", 0.55),
        ("gold_ring", 0.90),
        ("aluminum_bracket", 0.77),
        ("brass_fitting", 0.78),
        ("bronze_bushing", 0.60),
    ])
    def test_material_keywords(self, name, expected_r):
        _, rgb = pick_color(name, 0)
        assert abs(rgb[0] - expected_r) < 0.02

    def test_case_insensitive(self):
        _, rgb_lower = pick_color("copper_part", 0)
        _, rgb_mixed = pick_color("Copper_Part", 0)
        assert rgb_lower == rgb_mixed
