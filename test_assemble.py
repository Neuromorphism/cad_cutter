"""
Tests for the CAD Assembly Pipeline (assemble.py).

Covers: name parsing, concentric stacking, multi-level spanning parts,
cutting, STEP export, tessellation, mixed format support, and rendering.

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
    get_tight_bounding_box,
    shape_extent,
    translate_shape,
    scale_shape,
    apply_location,
    load_part,
    pick_color,
    parse_part_name,
    stack_parts,
    make_cutter,
    make_segment_cutter,
    cut_assembly,
    cut_part_direct,
    cut_part_direct_segment,
    _cutter_params,
    export_assembly_step,
    export_shape_step,
    export_shape,
    tessellate_shape,
    expand_inputs,
    orient_to_cylinder,
    _mesh_shell_to_solid,
    autoscale_parts,
    _parse_target_diameter,
    _get_xy_diameter,
    AXIS_MAP,
    MATERIAL_COLORS,
)

from cadquery import Location, Vector, Color
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform


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
def stl_part(tmp_dir):
    """Create a simple STL mesh file."""
    path = os.path.join(tmp_dir, "mesh_part.stl")
    box = cq.Workplane("XY").box(15, 15, 8)
    cq.exporters.export(box, path, exportType="STL")
    return path


def _make_concentric_parts(tmp_dir):
    """Create a full set of concentric parts for two levels."""
    paths = {}

    # Level 1
    outer1 = cq.Workplane("XY").box(60, 60, 15)
    p = os.path.join(tmp_dir, "outer_1.step")
    cq.exporters.export(outer1, p)
    paths["outer_1"] = p

    mid1 = cq.Workplane("XY").circle(20).circle(15).extrude(15)
    p = os.path.join(tmp_dir, "mid_1.step")
    cq.exporters.export(mid1, p)
    paths["mid_1"] = p

    inner1 = cq.Workplane("XY").cylinder(15, 5)
    p = os.path.join(tmp_dir, "inner_1.step")
    cq.exporters.export(inner1, p)
    paths["inner_1"] = p

    # Level 2
    outer2 = cq.Workplane("XY").box(50, 50, 12)
    p = os.path.join(tmp_dir, "outer_2.step")
    cq.exporters.export(outer2, p)
    paths["outer_2"] = p

    mid2 = cq.Workplane("XY").circle(18).circle(12).extrude(12)
    p = os.path.join(tmp_dir, "mid_2.step")
    cq.exporters.export(mid2, p)
    paths["mid_2"] = p

    inner2 = cq.Workplane("XY").cylinder(12, 4)
    p = os.path.join(tmp_dir, "inner_2.step")
    cq.exporters.export(inner2, p)
    paths["inner_2"] = p

    return paths


@pytest.fixture
def concentric_parts(tmp_dir):
    return _make_concentric_parts(tmp_dir)


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
# Test: Glob expansion (expand_inputs)
# ============================================================================

class TestExpandInputs:
    """Test that glob patterns and mixed-case extensions are expanded."""

    @pytest.fixture
    def glob_dir(self, tmp_dir):
        """Create files with mixed-case extensions for glob testing."""
        files = {}
        box = cq.Workplane("XY").box(10, 10, 5)
        for name, ext in [("outer_1", ".STEP"), ("outer_2", ".step"),
                          ("inner_1", ".STL"), ("inner_2", ".stl")]:
            p = os.path.join(tmp_dir, name + ext)
            if ext.lower() in (".step", ".stp"):
                cq.exporters.export(box, p)
            else:
                cq.exporters.export(box, p, exportType="STL")
            files[name] = p
        return tmp_dir, files

    def test_literal_paths_kept(self, glob_dir):
        d, files = glob_dir
        result = expand_inputs([files["outer_1"], files["inner_2"]])
        assert len(result) == 2

    def test_glob_star_step_finds_both_cases(self, glob_dir):
        d, files = glob_dir
        result = expand_inputs([os.path.join(d, "*.STEP")])
        basenames = {os.path.basename(r) for r in result}
        assert "outer_1.STEP" in basenames
        assert "outer_2.step" in basenames

    def test_glob_star_stl_finds_both_cases(self, glob_dir):
        d, files = glob_dir
        result = expand_inputs([os.path.join(d, "*.stl")])
        basenames = {os.path.basename(r) for r in result}
        assert "inner_1.STL" in basenames
        assert "inner_2.stl" in basenames

    def test_mixed_globs_and_literals(self, glob_dir):
        d, files = glob_dir
        result = expand_inputs([
            files["outer_1"],            # literal
            os.path.join(d, "*.stl"),    # glob
        ])
        assert len(result) == 3  # 1 literal + 2 stl

    def test_deduplication(self, glob_dir):
        d, files = glob_dir
        result = expand_inputs([
            files["outer_1"],
            os.path.join(d, "*.STEP"),  # includes outer_1.STEP again
        ])
        # outer_1.STEP should appear only once
        count = sum(1 for r in result if "outer_1" in r)
        assert count == 1

    def test_nonexistent_literal_passed_through(self):
        result = expand_inputs(["/no/such/file.step"])
        assert len(result) == 1
        assert result[0] == "/no/such/file.step"

    def test_glob_no_matches_returns_empty(self, tmp_dir):
        result = expand_inputs([os.path.join(tmp_dir, "*.xyz")])
        assert len(result) == 0

    def test_non_cad_files_filtered(self, tmp_dir):
        # Create a .txt file in the directory
        txt = os.path.join(tmp_dir, "readme.txt")
        with open(txt, "w") as f:
            f.write("not a cad file")
        result = expand_inputs([os.path.join(tmp_dir, "*")])
        assert not any(r.endswith(".txt") for r in result)


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
        assert rgb1 != rgb2

    def test_palette_wraps(self):
        _, rgb0 = pick_color("part", 0)
        _, rgb8 = pick_color("part", 8)
        assert rgb0 == rgb8


# ============================================================================
# Test: Part name parsing
# ============================================================================

class TestParsePartName:
    def test_outer_1(self):
        tier, levels, _seg = parse_part_name("outer_1")
        assert tier == "outer"
        assert levels == [1]

    def test_mid_2(self):
        tier, levels, _seg = parse_part_name("mid_2")
        assert tier == "mid"
        assert levels == [2]

    def test_inner_3(self):
        tier, levels, _seg = parse_part_name("inner_3")
        assert tier == "inner"
        assert levels == [3]

    def test_case_insensitive(self):
        tier, levels, _seg = parse_part_name("OUTER_1")
        assert tier == "outer"
        assert levels == [1]

    def test_mixed_case(self):
        tier, levels, _seg = parse_part_name("Inner_5")
        assert tier == "inner"
        assert levels == [5]

    def test_hyphen_separator(self):
        tier, levels, _seg = parse_part_name("outer-1")
        assert tier == "outer"
        assert levels == [1]

    def test_no_separator(self):
        tier, levels, _seg = parse_part_name("outer1")
        assert tier == "outer"
        assert levels == [1]

    def test_with_material_prefix(self):
        tier, levels, _seg = parse_part_name("steel_outer_2")
        assert tier == "outer"
        assert levels == [2]

    def test_unrecognized_name(self):
        tier, levels, _seg = parse_part_name("plate")
        assert tier is None
        assert levels is None

    def test_unrecognized_with_number(self):
        tier, levels, _seg = parse_part_name("flange_3")
        assert tier is None
        assert levels is None

    def test_multi_digit_level(self):
        tier, levels, _seg = parse_part_name("outer_12")
        assert tier == "outer"
        assert levels == [12]

    def test_multi_level_span(self):
        tier, levels, _seg = parse_part_name("inner_2_3")
        assert tier == "inner"
        assert levels == [2, 3]

    def test_multi_level_three_sections(self):
        tier, levels, _seg = parse_part_name("mid_1_2_3")
        assert tier == "mid"
        assert levels == [1, 2, 3]

    def test_multi_level_with_material(self):
        tier, levels, _seg = parse_part_name("inner_2_3_steel")
        assert tier == "inner"
        assert levels == [2, 3]

    def test_multi_level_hyphen(self):
        tier, levels, _seg = parse_part_name("outer-1-2")
        assert tier == "outer"
        assert levels == [1, 2]


# ============================================================================
# Test: Concentric stacking
# ============================================================================

class TestConcentricStacking:
    def test_single_outer_base_at_z0(self, tmp_dir):
        """A single outer_1 part should have its base at z=0."""
        path = os.path.join(tmp_dir, "outer_1.step")
        box = cq.Workplane("XY").box(40, 40, 10)
        cq.exporters.export(box, path)

        wp, name = load_part(path)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        assert len(info) == 1

        moved = apply_location(info[0][1], info[0][2])
        _, _, zn, _, _, zx = get_bounding_box(moved)
        assert abs(zn - 0) < 0.1
        assert abs(zx - 10) < 0.1

    def test_two_outer_levels_stack(self, tmp_dir):
        """outer_1 and outer_2 should stack vertically."""
        p1 = os.path.join(tmp_dir, "outer_1.step")
        cq.exporters.export(cq.Workplane("XY").box(40, 40, 10), p1)

        p2 = os.path.join(tmp_dir, "outer_2.step")
        cq.exporters.export(cq.Workplane("XY").box(30, 30, 8), p2)

        parts = [load_part(p) for p in [p1, p2]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        assert len(info) == 2

        # outer_1: z = [0, 10]
        m1 = apply_location(info[0][1], info[0][2])
        _, _, z1n, _, _, z1x = get_bounding_box(m1)
        assert abs(z1n - 0) < 0.1
        assert abs(z1x - 10) < 0.1

        # outer_2: z = [10, 18]
        m2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, z2x = get_bounding_box(m2)
        assert abs(z2n - 10) < 0.1
        assert abs(z2x - 18) < 0.1

    def test_gap_between_outer_levels(self, tmp_dir):
        """Gap should separate outer levels."""
        p1 = os.path.join(tmp_dir, "outer_1.step")
        cq.exporters.export(cq.Workplane("XY").box(40, 40, 10), p1)

        p2 = os.path.join(tmp_dir, "outer_2.step")
        cq.exporters.export(cq.Workplane("XY").box(30, 30, 8), p2)

        parts = [load_part(p) for p in [p1, p2]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=5)

        m1 = apply_location(info[0][1], info[0][2])
        _, _, _, _, _, z1x = get_bounding_box(m1)

        m2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, _ = get_bounding_box(m2)

        assert abs(z2n - z1x - 5) < 0.5

    def test_mid_centered_in_outer_xy(self, concentric_parts):
        """mid_1 should be XY-centered within outer_1."""
        parts = [
            load_part(concentric_parts["outer_1"]),
            load_part(concentric_parts["mid_1"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        # Find mid and outer in results
        outer_info = [i for i in info if "outer" in i[0]][0]
        mid_info = [i for i in info if "mid" in i[0]][0]

        outer_moved = apply_location(outer_info[1], outer_info[2])
        mid_moved = apply_location(mid_info[1], mid_info[2])

        oxn, oyn, _, oxx, oyx, _ = get_bounding_box(outer_moved)
        mxn, myn, _, mxx, myx, _ = get_bounding_box(mid_moved)

        outer_cx = (oxn + oxx) / 2
        outer_cy = (oyn + oyx) / 2
        mid_cx = (mxn + mxx) / 2
        mid_cy = (myn + myx) / 2

        assert abs(mid_cx - outer_cx) < 0.5
        assert abs(mid_cy - outer_cy) < 0.5

    def test_inner_centered_in_mid_xy(self, concentric_parts):
        """inner_1 should be XY-centered within mid_1."""
        parts = [
            load_part(concentric_parts["outer_1"]),
            load_part(concentric_parts["mid_1"]),
            load_part(concentric_parts["inner_1"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        mid_info = [i for i in info if "mid" in i[0]][0]
        inner_info = [i for i in info if "inner" in i[0]][0]

        mid_moved = apply_location(mid_info[1], mid_info[2])
        inner_moved = apply_location(inner_info[1], inner_info[2])

        mxn, myn, _, mxx, myx, _ = get_bounding_box(mid_moved)
        ixn, iyn, _, ixx, iyx, _ = get_bounding_box(inner_moved)

        mid_cx = (mxn + mxx) / 2
        mid_cy = (myn + myx) / 2
        inner_cx = (ixn + ixx) / 2
        inner_cy = (iyn + iyx) / 2

        assert abs(inner_cx - mid_cx) < 0.5
        assert abs(inner_cy - mid_cy) < 0.5

    def test_same_level_same_z_base(self, concentric_parts):
        """All tiers at the same level should share the same Z base."""
        parts = [
            load_part(concentric_parts["outer_1"]),
            load_part(concentric_parts["mid_1"]),
            load_part(concentric_parts["inner_1"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        z_bottoms = []
        for name, shape, loc, rgb, *_rest in info:
            moved = apply_location(shape, loc)
            _, _, zn, _, _, _ = get_bounding_box(moved)
            z_bottoms.append(zn)

        # All parts at level 1 should start at z=0
        for zn in z_bottoms:
            assert abs(zn - 0) < 0.5

    def test_two_levels_full_concentric(self, concentric_parts):
        """Full 2-level concentric assembly."""
        parts = [
            load_part(concentric_parts["outer_1"]),
            load_part(concentric_parts["mid_1"]),
            load_part(concentric_parts["inner_1"]),
            load_part(concentric_parts["outer_2"]),
            load_part(concentric_parts["mid_2"]),
            load_part(concentric_parts["inner_2"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=2)
        assert len(info) == 6

        # Level 1 parts at z=0, level 2 parts at z=15+2=17
        level1 = [i for i in info if "_1" in i[0]]
        level2 = [i for i in info if "_2" in i[0]]

        for name, shape, loc, rgb, *_rest in level1:
            moved = apply_location(shape, loc)
            _, _, zn, _, _, _ = get_bounding_box(moved)
            assert abs(zn - 0) < 0.5, f"{name} z_bottom={zn}, expected 0"

        for name, shape, loc, rgb, *_rest in level2:
            moved = apply_location(shape, loc)
            _, _, zn, _, _, _ = get_bounding_box(moved)
            assert abs(zn - 17) < 0.5, f"{name} z_bottom={zn}, expected 17"

    def test_input_order_doesnt_matter(self, concentric_parts):
        """Parts can be given in any order — result should be the same."""
        # Forward order
        parts_fwd = [
            load_part(concentric_parts["outer_1"]),
            load_part(concentric_parts["mid_1"]),
            load_part(concentric_parts["inner_1"]),
        ]
        _, info_fwd = stack_parts(parts_fwd, AXIS_MAP["z"], gap=0)

        # Reverse order
        parts_rev = [
            load_part(concentric_parts["inner_1"]),
            load_part(concentric_parts["mid_1"]),
            load_part(concentric_parts["outer_1"]),
        ]
        _, info_rev = stack_parts(parts_rev, AXIS_MAP["z"], gap=0)

        # Both should produce 3 parts with same Z bottoms
        def get_z_map(info):
            return {n: get_bounding_box(apply_location(s, l))[2]
                    for n, s, l, *_rest in info}

        fwd_z = get_z_map(info_fwd)
        rev_z = get_z_map(info_rev)

        for name in fwd_z:
            assert abs(fwd_z[name] - rev_z[name]) < 0.5


class TestSpanningParts:
    """Parts that span multiple sections (e.g. inner_2_3_steel)."""

    @pytest.fixture
    def three_level_parts(self, tmp_dir):
        """Create 3 outer levels + a spanning inner part."""
        paths = {}
        for i in range(1, 4):
            p = os.path.join(tmp_dir, f"outer_{i}.step")
            box = cq.Workplane("XY").box(40, 40, 10)
            cq.exporters.export(box, p)
            paths[f"outer_{i}"] = p

        # Inner part spanning levels 2-3, height 20 (should cover 2 levels)
        p = os.path.join(tmp_dir, "inner_2_3_steel.step")
        cyl = cq.Workplane("XY").cylinder(20, 5)
        cq.exporters.export(cyl, p)
        paths["inner_2_3_steel"] = p

        return paths

    def test_spanning_part_name_parsed(self):
        tier, levels, _seg = parse_part_name("inner_2_3_steel")
        assert tier == "inner"
        assert levels == [2, 3]

    def test_spanning_part_z_base(self, three_level_parts):
        """A spanning part's Z-base should align with the first spanned level."""
        parts = []
        for key in ["outer_1", "outer_2", "outer_3", "inner_2_3_steel"]:
            parts.append(load_part(three_level_parts[key]))

        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        # Find the spanning part and level-2 outer
        span_info = None
        lvl2_info = None
        for name, shape, loc, rgb, *_rest in info:
            if "2_3" in name:
                span_info = (name, shape, loc, rgb)
            elif name == "outer_2":
                lvl2_info = (name, shape, loc, rgb)

        assert span_info is not None, "Spanning part not found in assembly"
        assert lvl2_info is not None, "outer_2 not found in assembly"

        # Spanning part's Z-base should match outer_2's Z-base
        span_moved = apply_location(span_info[1], span_info[2])
        lvl2_moved = apply_location(lvl2_info[1], lvl2_info[2])
        _, _, span_zn, _, _, _ = get_bounding_box(span_moved)
        _, _, lvl2_zn, _, _, _ = get_bounding_box(lvl2_moved)
        assert abs(span_zn - lvl2_zn) < 0.5

    def test_spanning_part_is_xy_centered(self, three_level_parts):
        """Spanning part should be XY-centered at origin."""
        parts = []
        for key in ["outer_1", "outer_2", "outer_3", "inner_2_3_steel"]:
            parts.append(load_part(three_level_parts[key]))

        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        for name, shape, loc, rgb, *_rest in info:
            if "2_3" in name:
                moved = apply_location(shape, loc)
                xn, yn, _, xx, yx, _ = get_bounding_box(moved)
                cx = (xn + xx) / 2.0
                cy = (yn + yx) / 2.0
                assert abs(cx) < 0.5, f"Spanning part not centered in X: {cx}"
                assert abs(cy) < 0.5, f"Spanning part not centered in Y: {cy}"

    def test_spanning_part_with_gap(self, three_level_parts):
        """Spanning part should respect gaps between levels."""
        parts = []
        for key in ["outer_1", "outer_2", "outer_3", "inner_2_3_steel"]:
            parts.append(load_part(three_level_parts[key]))

        gap = 5.0
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=gap)

        # Level 1 outer is 10 high, gap 5 → level 2 starts at 15
        for name, shape, loc, rgb, *_rest in info:
            if "2_3" in name:
                moved = apply_location(shape, loc)
                _, _, span_zn, _, _, _ = get_bounding_box(moved)
                assert abs(span_zn - 15.0) < 0.5

    def test_spanning_part_total_count(self, three_level_parts):
        """Assembly should contain all 4 parts."""
        parts = []
        for key in ["outer_1", "outer_2", "outer_3", "inner_2_3_steel"]:
            parts.append(load_part(three_level_parts[key]))

        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        assert len(info) == 4

    def test_spanning_part_gets_material_color(self, three_level_parts):
        """inner_2_3_steel should be colored as steel."""
        parts = []
        for key in ["outer_1", "outer_2", "outer_3", "inner_2_3_steel"]:
            parts.append(load_part(three_level_parts[key]))

        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        for name, shape, loc, rgb, *_rest in info:
            if "steel" in name.lower():
                # Steel color: R ~ 0.55
                assert abs(rgb[0] - 0.55) < 0.01, f"Expected steel color, got {rgb}"

    def test_spanning_three_levels(self, tmp_dir):
        """A part spanning levels 1-2-3 starts at level 1."""
        parts = []
        for i in range(1, 4):
            p = os.path.join(tmp_dir, f"outer_{i}.step")
            cq.Workplane("XY").box(40, 40, 10).val().exportStep(p)
            parts.append(load_part(p))

        span_path = os.path.join(tmp_dir, "mid_1_2_3.step")
        cq.Workplane("XY").cylinder(30, 8).val().exportStep(span_path)
        parts.append(load_part(span_path))

        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        for name, shape, loc, rgb, *_rest in info:
            if "1_2_3" in name:
                moved = apply_location(shape, loc)
                _, _, zn, _, _, _ = get_bounding_box(moved)
                assert abs(zn) < 0.5, "3-level span should start at z=0"

    def test_spanning_part_cuttable(self, three_level_parts):
        """A spanning part should be cuttable like any other part."""
        parts = []
        for key in ["outer_1", "outer_2", "inner_2_3_steel"]:
            parts.append(load_part(three_level_parts[key]))

        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        # Cut each part
        for name, shape, loc, rgb, *_rest in info:
            moved = apply_location(shape, loc)
            bbox = Bnd_Box()
            BRepBndLib.Add_s(moved, bbox)
            cutter = make_cutter(bbox.Get(), 90, AXIS_MAP["z"])
            result = cut_assembly(moved, cutter)
            assert not result.IsNull(), f"Cut failed for spanning part '{name}'"


class TestLegacyStacking:
    """Parts without tier/level names stack as sequential outer parts."""

    def test_unnamed_parts_stack_sequentially(self, box_step, cylinder_step):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["z"], gap=0)
        assert len(info) == 2

        m1 = apply_location(info[0][1], info[0][2])
        _, _, z1n, _, _, z1x = get_bounding_box(m1)
        assert abs(z1n - 0) < 0.1

        m2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, _ = get_bounding_box(m2)
        assert abs(z2n - z1x) < 0.5

    def test_unnamed_parts_gap(self, box_step, cylinder_step):
        wp1, n1 = load_part(box_step)
        wp2, n2 = load_part(cylinder_step)
        gap = 5.0
        assy, info = stack_parts([(wp1, n1), (wp2, n2)], AXIS_MAP["z"], gap=gap)

        m1 = apply_location(info[0][1], info[0][2])
        _, _, _, _, _, z1x = get_bounding_box(m1)

        m2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, _ = get_bounding_box(m2)

        assert abs(z2n - z1x - gap) < 0.5

    def test_three_unnamed_parts_ordered(self, box_step, cylinder_step, flange_step):
        parts = [load_part(p) for p in [box_step, cylinder_step, flange_step]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=2)
        assert len(info) == 3

        prev_top = None
        for name, shape, loc, rgb, *_rest in info:
            moved = apply_location(shape, loc)
            _, _, zn, _, _, zx = get_bounding_box(moved)
            if prev_top is not None:
                assert zn >= prev_top + 1.5
            prev_top = zx


# ============================================================================
# Test: Cutting
# ============================================================================

class TestCutting:
    def _build_compound(self, info):
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for name, shape, loc, rgb, *_rest in info:
            moved = apply_location(shape, loc)
            builder.Add(compound, moved)
        return compound

    def test_cut_90_removes_quarter(self, box_step):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        compound = self._build_compound(info)

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 90, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)

        assert not result.IsNull()
        result_bbox = Bnd_Box()
        BRepBndLib.Add_s(result, result_bbox)
        assert not result_bbox.IsVoid()

    def test_cut_180_removes_half(self, box_step):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        compound = self._build_compound(info)

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 180, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)
        assert not result.IsNull()

    def test_cut_concentric_assembly(self, concentric_parts):
        """Cut a full concentric assembly."""
        parts = [
            load_part(concentric_parts["outer_1"]),
            load_part(concentric_parts["mid_1"]),
            load_part(concentric_parts["inner_1"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        compound = self._build_compound(info)

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 90, AXIS_MAP["z"])
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
        for n, s, l, c, *_rest in info:
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

        reimported = cq.importers.importStep(out_path)
        shape = reimported.val().wrapped
        assert not shape.IsNull()

    def test_shape_export_stl_is_reimportable(self, box_step, tmp_dir):
        """Export STL from a shape and ensure it can be loaded back."""
        wp, _name = load_part(box_step)
        shape = wp.val().wrapped

        out_path = os.path.join(tmp_dir, "shape_out.stl")
        export_shape(shape, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 100

        reimported_wp, _ = load_part(out_path)
        re_shape = reimported_wp.val().wrapped
        assert not re_shape.IsNull()

    def test_concentric_export_reimport(self, concentric_parts, tmp_dir):
        """Export a concentric assembly and reimport."""
        parts = [
            load_part(concentric_parts["outer_1"]),
            load_part(concentric_parts["mid_1"]),
            load_part(concentric_parts["inner_1"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        out_path = os.path.join(tmp_dir, "concentric_out.step")
        export_assembly_step(assy, out_path)
        assert os.path.exists(out_path)

        reimported = cq.importers.importStep(out_path)
        assert not reimported.val().wrapped.IsNull()


# ============================================================================
# Test: Tessellation
# ============================================================================

class TestTessellation:
    def test_box_tessellation(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        verts, faces = tessellate_shape(box.val().wrapped, tolerance=0.1)
        assert len(verts) > 0
        assert len(faces) > 0
        assert len(verts) >= 8
        assert len(faces) >= 12

    def test_cylinder_tessellation(self):
        cyl = cq.Workplane("XY").cylinder(20, 5)
        verts, faces = tessellate_shape(cyl.val().wrapped, tolerance=0.1)
        assert len(verts) > 20
        assert len(faces) > 10

    def test_face_format(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        verts, faces = tessellate_shape(box.val().wrapped, tolerance=0.1)
        for face in faces:
            assert face[0] == 3
            assert all(0 <= face[j] < len(verts) for j in range(1, 4))

    def test_tolerance_affects_detail(self):
        cyl = cq.Workplane("XY").cylinder(20, 5)
        shape = cyl.val().wrapped
        v_coarse, f_coarse = tessellate_shape(shape, tolerance=1.0)
        v_fine, f_fine = tessellate_shape(shape, tolerance=0.01)
        assert len(v_fine) >= len(v_coarse)


# ============================================================================
# Test: End-to-end pipeline (no rendering)
# ============================================================================

class TestPipelineNoRender:
    def test_single_part_pipeline(self, box_step, tmp_dir):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        out = os.path.join(tmp_dir, "single.step")
        export_assembly_step(assy, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100

    def test_concentric_pipeline(self, concentric_parts, tmp_dir):
        """Full concentric assembly pipeline."""
        parts = [load_part(concentric_parts[k])
                 for k in ["outer_1", "mid_1", "inner_1", "outer_2", "mid_2", "inner_2"]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=2)
        out = os.path.join(tmp_dir, "concentric.step")
        export_assembly_step(assy, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100

    def test_cut_concentric_pipeline(self, concentric_parts, tmp_dir):
        """Cut a concentric assembly and export."""
        parts = [load_part(concentric_parts[k])
                 for k in ["outer_1", "mid_1", "inner_1"]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for n, s, l, c, *_rest in info:
            builder.Add(compound, apply_location(s, l))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 120, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)

        out = os.path.join(tmp_dir, "cut_concentric.step")
        export_shape_step(result, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100

    def test_mesh_input_pipeline(self, stl_part, tmp_dir):
        wp, name = load_part(stl_part)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)
        assert len(info) == 1

    def test_mixed_cad_mesh_pipeline(self, box_step, stl_part, tmp_dir):
        parts = [load_part(p) for p in [box_step, stl_part]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=2)
        assert len(info) == 2


# ============================================================================
# Test: Edge cases
# ============================================================================

class TestEdgeCases:
    def test_very_small_angle(self, box_step):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for n, s, l, c, *_rest in info:
            builder.Add(compound, apply_location(s, l))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 5, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)
        assert not result.IsNull()

    def test_near_full_angle(self, box_step):
        wp, name = load_part(box_step)
        assy, info = stack_parts([(wp, name)], AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for n, s, l, c, *_rest in info:
            builder.Add(compound, apply_location(s, l))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        cutter = make_cutter(bbox.Get(), 350, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)
        assert not result.IsNull()

    def test_zero_gap(self, tmp_dir):
        p1 = os.path.join(tmp_dir, "outer_1.step")
        cq.exporters.export(cq.Workplane("XY").box(20, 20, 10), p1)
        p2 = os.path.join(tmp_dir, "outer_2.step")
        cq.exporters.export(cq.Workplane("XY").box(20, 20, 8), p2)

        parts = [load_part(p) for p in [p1, p2]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        m1 = apply_location(info[0][1], info[0][2])
        _, _, _, _, _, z1x = get_bounding_box(m1)

        m2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, _ = get_bounding_box(m2)

        assert abs(z2n - z1x) < 0.5

    def test_large_gap(self, tmp_dir):
        p1 = os.path.join(tmp_dir, "outer_1.step")
        cq.exporters.export(cq.Workplane("XY").box(20, 20, 10), p1)
        p2 = os.path.join(tmp_dir, "outer_2.step")
        cq.exporters.export(cq.Workplane("XY").box(20, 20, 8), p2)

        parts = [load_part(p) for p in [p1, p2]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=100)

        m1 = apply_location(info[0][1], info[0][2])
        _, _, _, _, _, z1x = get_bounding_box(m1)

        m2 = apply_location(info[1][1], info[1][2])
        _, _, z2n, _, _, _ = get_bounding_box(m2)

        assert abs(z2n - z1x - 100) < 0.5

    def test_only_mid_and_inner_no_outer(self, tmp_dir):
        """Level with mid and inner but no outer should still work."""
        p1 = os.path.join(tmp_dir, "mid_1.step")
        cq.exporters.export(cq.Workplane("XY").circle(15).extrude(10), p1)
        p2 = os.path.join(tmp_dir, "inner_1.step")
        cq.exporters.export(cq.Workplane("XY").cylinder(10, 3), p2)

        parts = [load_part(p) for p in [p1, p2]]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        assert len(info) == 2


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


# ============================================================================
# Test: Mixed format loading and assembly
# ============================================================================

class TestMixedFormats:
    """Test that .step, .stl, .obj, and .ply files can be mixed freely."""

    @pytest.fixture
    def mixed_parts(self, tmp_dir):
        """Create parts in multiple formats with tier/level names."""
        paths = {}

        # outer_1 as STEP
        outer1 = cq.Workplane("XY").box(40, 40, 10)
        p = os.path.join(tmp_dir, "outer_1.step")
        cq.exporters.export(outer1, p)
        paths["outer_1.step"] = p

        # inner_1 as STL
        inner1 = cq.Workplane("XY").cylinder(10, 4)
        p = os.path.join(tmp_dir, "inner_1.stl")
        cq.exporters.export(inner1, p, exportType="STL")
        paths["inner_1.stl"] = p

        # outer_2 as STEP
        outer2 = cq.Workplane("XY").box(35, 35, 8)
        p = os.path.join(tmp_dir, "outer_2.step")
        cq.exporters.export(outer2, p)
        paths["outer_2.step"] = p

        # inner_2 as STL
        inner2 = cq.Workplane("XY").cylinder(8, 3)
        p = os.path.join(tmp_dir, "inner_2.stl")
        cq.exporters.export(inner2, p, exportType="STL")
        paths["inner_2.stl"] = p

        return paths

    def test_load_step(self, mixed_parts):
        wp, name = load_part(mixed_parts["outer_1.step"])
        assert name == "outer_1"
        assert not wp.val().wrapped.IsNull()

    def test_load_stl(self, mixed_parts):
        wp, name = load_part(mixed_parts["inner_1.stl"])
        assert name == "inner_1"
        assert not wp.val().wrapped.IsNull()

    def test_mixed_step_stl_assembly(self, mixed_parts):
        """Assemble STEP + STL parts concentrically."""
        parts = [
            load_part(mixed_parts["outer_1.step"]),
            load_part(mixed_parts["inner_1.stl"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        assert len(info) == 2

        # Both should have z-base at 0 (same level)
        for name, shape, loc, rgb, *_rest in info:
            moved = apply_location(shape, loc)
            _, _, zn, _, _, _ = get_bounding_box(moved)
            assert abs(zn - 0) < 0.5

    def test_mixed_two_level_assembly(self, mixed_parts):
        """Two-level assembly mixing STEP and STL."""
        parts = [
            load_part(mixed_parts["outer_1.step"]),
            load_part(mixed_parts["inner_1.stl"]),
            load_part(mixed_parts["outer_2.step"]),
            load_part(mixed_parts["inner_2.stl"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=1)
        assert len(info) == 4

    def test_mixed_format_cut(self, mixed_parts):
        """Cut a mixed STEP+STL assembly."""
        parts = [
            load_part(mixed_parts["outer_1.step"]),
            load_part(mixed_parts["inner_1.stl"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        # Cut each part individually (as the pipeline does)
        cut_ok = 0
        for name, shape, loc, rgb, *_rest in info:
            moved = apply_location(shape, loc)
            bbox = Bnd_Box()
            BRepBndLib.Add_s(moved, bbox)
            cutter = make_cutter(bbox.Get(), 90, AXIS_MAP["z"])
            result = cut_assembly(moved, cutter)
            assert not result.IsNull()
            cut_ok += 1
        assert cut_ok == 2

    def test_mixed_format_step_export(self, mixed_parts, tmp_dir):
        """Export mixed-format assembly to STEP."""
        parts = [
            load_part(mixed_parts["outer_1.step"]),
            load_part(mixed_parts["inner_1.stl"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        out = os.path.join(tmp_dir, "mixed.step")
        export_assembly_step(assy, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 100

    def test_mixed_format_tessellation(self, mixed_parts):
        """Tessellate parts from different formats."""
        for key in mixed_parts:
            wp, name = load_part(mixed_parts[key])
            verts, faces = tessellate_shape(wp.val().wrapped, tolerance=0.1)
            assert len(verts) > 0, f"No vertices for {key}"
            assert len(faces) > 0, f"No faces for {key}"


class TestMeshToSolid:
    """Test mesh-to-solid conversion for boolean compatibility."""

    def test_stl_becomes_cuttable(self, tmp_dir):
        """An STL-loaded part should survive a boolean cut."""
        p = os.path.join(tmp_dir, "box.stl")
        box = cq.Workplane("XY").box(20, 20, 10)
        cq.exporters.export(box, p, exportType="STL")

        wp, name = load_part(p)
        shape = wp.val().wrapped

        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        cutter = make_cutter(bbox.Get(), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        assert not result.IsNull()

    def test_solid_conversion_preserves_geometry(self, tmp_dir):
        """Mesh-to-solid should keep roughly the same bounding box."""
        p = os.path.join(tmp_dir, "cyl.stl")
        cyl = cq.Workplane("XY").cylinder(20, 8)
        cq.exporters.export(cyl, p, exportType="STL")

        wp, _ = load_part(p)
        shape = wp.val().wrapped
        xn, yn, zn, xx, yx, zx = get_bounding_box(shape)

        # Cylinder: radius 8, height 20 → x/y range ~[-8,8], z range ~[-10,10]
        assert abs((xx - xn) - 16) < 1.0
        assert abs((zx - zn) - 20) < 1.0


# ============================================================================
# Test: Segment splitting (a/b suffix)
# ============================================================================

class TestSegmentParsing:
    """Test that parse_part_name recognises the a/b segment suffix."""

    def test_inner_2a(self):
        tier, levels, segment = parse_part_name("inner_2a")
        assert tier == "inner"
        assert levels == [2]
        assert segment == "a"

    def test_inner_2b(self):
        tier, levels, segment = parse_part_name("inner_2b")
        assert tier == "inner"
        assert levels == [2]
        assert segment == "b"

    def test_no_segment(self):
        tier, levels, segment = parse_part_name("inner_2")
        assert tier == "inner"
        assert levels == [2]
        assert segment is None

    def test_outer_1a(self):
        tier, levels, segment = parse_part_name("outer_1a")
        assert tier == "outer"
        assert levels == [1]
        assert segment == "a"

    def test_mid_3b(self):
        tier, levels, segment = parse_part_name("mid_3b")
        assert tier == "mid"
        assert levels == [3]
        assert segment == "b"

    def test_case_insensitive_segment(self):
        tier, levels, segment = parse_part_name("Inner_2A")
        assert tier == "inner"
        assert levels == [2]
        assert segment == "a"

    def test_segment_with_material(self):
        """inner_2a_steel should parse segment='a' from the level number."""
        tier, levels, segment = parse_part_name("inner_2a_steel")
        assert tier == "inner"
        assert levels == [2]
        assert segment == "a"

    def test_multi_level_with_segment(self):
        """inner_2_3a should parse levels [2,3] with segment 'a'."""
        tier, levels, segment = parse_part_name("inner_2_3a")
        assert tier == "inner"
        assert levels == [2, 3]
        assert segment == "a"


class TestSegmentStacking:
    """Test that a/b parts are placed at the same position."""

    @pytest.fixture
    def segment_parts(self, tmp_dir):
        paths = {}
        outer = cq.Workplane("XY").box(60, 60, 15)
        p = os.path.join(tmp_dir, "outer_2.step")
        cq.exporters.export(outer, p)
        paths["outer_2"] = p

        inner_a = cq.Workplane("XY").cylinder(15, 5)
        p = os.path.join(tmp_dir, "inner_2a.step")
        cq.exporters.export(inner_a, p)
        paths["inner_2a"] = p

        inner_b = cq.Workplane("XY").cylinder(15, 5)
        p = os.path.join(tmp_dir, "inner_2b.step")
        cq.exporters.export(inner_b, p)
        paths["inner_2b"] = p

        return paths

    def test_both_segments_in_assembly(self, segment_parts):
        """Both inner_2a and inner_2b should appear in the assembly."""
        parts = [
            load_part(segment_parts["outer_2"]),
            load_part(segment_parts["inner_2a"]),
            load_part(segment_parts["inner_2b"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        names = [i[0] for i in info]
        assert "inner_2a" in names
        assert "inner_2b" in names
        assert len(info) == 3

    def test_segments_same_z_base(self, segment_parts):
        """inner_2a and inner_2b should have the same Z base."""
        parts = [
            load_part(segment_parts["outer_2"]),
            load_part(segment_parts["inner_2a"]),
            load_part(segment_parts["inner_2b"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        z_bases = {}
        for name, shape, loc, rgb, seg in info:
            moved = apply_location(shape, loc)
            _, _, zn, _, _, _ = get_bounding_box(moved)
            z_bases[name] = zn

        assert abs(z_bases["inner_2a"] - z_bases["inner_2b"]) < 0.1

    def test_segments_xy_centered(self, segment_parts):
        """inner_2a and inner_2b should be XY-centered at origin."""
        parts = [
            load_part(segment_parts["outer_2"]),
            load_part(segment_parts["inner_2a"]),
            load_part(segment_parts["inner_2b"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        for name, shape, loc, rgb, seg in info:
            if seg is not None:
                moved = apply_location(shape, loc)
                xn, yn, _, xx, yx, _ = get_bounding_box(moved)
                cx = (xn + xx) / 2.0
                cy = (yn + yx) / 2.0
                assert abs(cx) < 0.5, f"{name} not X-centered: {cx}"
                assert abs(cy) < 0.5, f"{name} not Y-centered: {cy}"

    def test_segment_info_propagated(self, segment_parts):
        """The segment field should be propagated in part_info."""
        parts = [
            load_part(segment_parts["outer_2"]),
            load_part(segment_parts["inner_2a"]),
            load_part(segment_parts["inner_2b"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        segments = {i[0]: i[4] for i in info}
        assert segments["inner_2a"] == "a"
        assert segments["inner_2b"] == "b"
        assert segments["outer_2"] is None


class TestSegmentCutting:
    """Test segment-aware cutting with a/b halves."""

    @pytest.fixture
    def segment_assembly(self, tmp_dir):
        """Create an assembly with outer_1, inner_1a, inner_1b."""
        paths = {}
        outer = cq.Workplane("XY").box(60, 60, 15)
        p = os.path.join(tmp_dir, "outer_1.step")
        cq.exporters.export(outer, p)
        paths["outer_1"] = p

        inner_a = cq.Workplane("XY").cylinder(15, 8)
        p = os.path.join(tmp_dir, "inner_1a.step")
        cq.exporters.export(inner_a, p)
        paths["inner_1a"] = p

        inner_b = cq.Workplane("XY").cylinder(15, 8)
        p = os.path.join(tmp_dir, "inner_1b.step")
        cq.exporters.export(inner_b, p)
        paths["inner_1b"] = p

        return paths

    def test_segment_cutter_a_not_null(self, segment_assembly):
        """Cutting inner_1a with the 'a' segment cutter should not be null."""
        parts = [
            load_part(segment_assembly["outer_1"]),
            load_part(segment_assembly["inner_1a"]),
            load_part(segment_assembly["inner_1b"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        # Build compound for bbox
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for name, shape, loc, rgb, seg in info:
            builder.Add(compound, apply_location(shape, loc))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()

        cut_angle = 120.0
        cutter_a = make_segment_cutter(bbox_vals, cut_angle, AXIS_MAP["z"], "a")

        # Cut the inner_1a part
        for name, shape, loc, rgb, seg in info:
            if seg == "a":
                moved = apply_location(shape, loc)
                result = cut_assembly(moved, cutter_a)
                assert not result.IsNull()

    def test_segment_cutter_b_not_null(self, segment_assembly):
        """Cutting inner_1b with the 'b' segment cutter should not be null."""
        parts = [
            load_part(segment_assembly["outer_1"]),
            load_part(segment_assembly["inner_1a"]),
            load_part(segment_assembly["inner_1b"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for name, shape, loc, rgb, seg in info:
            builder.Add(compound, apply_location(shape, loc))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()

        cut_angle = 120.0
        cutter_b = make_segment_cutter(bbox_vals, cut_angle, AXIS_MAP["z"], "b")

        for name, shape, loc, rgb, seg in info:
            if seg == "b":
                moved = apply_location(shape, loc)
                result = cut_assembly(moved, cutter_b)
                assert not result.IsNull()

    def test_a_and_b_halves_dont_overlap(self, segment_assembly):
        """The 'a' and 'b' cut results should have non-overlapping bounding boxes."""
        parts = [
            load_part(segment_assembly["outer_1"]),
            load_part(segment_assembly["inner_1a"]),
            load_part(segment_assembly["inner_1b"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for name, shape, loc, rgb, seg in info:
            builder.Add(compound, apply_location(shape, loc))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()

        cut_angle = 120.0
        cutter_a = make_segment_cutter(bbox_vals, cut_angle, AXIS_MAP["z"], "a")
        cutter_b = make_segment_cutter(bbox_vals, cut_angle, AXIS_MAP["z"], "b")

        # Use the inner_1a part for both cuts (same geometry)
        inner_shape = None
        for name, shape, loc, rgb, seg in info:
            if seg == "a":
                inner_shape = apply_location(shape, loc)
                break

        result_a = cut_assembly(inner_shape, cutter_a)
        result_b = cut_assembly(inner_shape, cutter_b)

        # Both should be non-void
        bbox_a = Bnd_Box()
        BRepBndLib.Add_s(result_a, bbox_a)
        assert not bbox_a.IsVoid()

        bbox_b = Bnd_Box()
        BRepBndLib.Add_s(result_b, bbox_b)
        assert not bbox_b.IsVoid()

    def test_only_a_specified_b_side_empty(self, tmp_dir):
        """If only 'a' is specified, the 'b' side should be empty."""
        outer = cq.Workplane("XY").box(60, 60, 15)
        p_outer = os.path.join(tmp_dir, "outer_1.step")
        cq.exporters.export(outer, p_outer)

        inner_a = cq.Workplane("XY").cylinder(15, 8)
        p_inner_a = os.path.join(tmp_dir, "inner_1a.step")
        cq.exporters.export(inner_a, p_inner_a)

        parts = [load_part(p_outer), load_part(p_inner_a)]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        # Only 'a' segment, no 'b' — assembly should have 2 parts
        assert len(info) == 2
        segments = {i[0]: i[4] for i in info}
        assert segments["inner_1a"] == "a"
        assert "inner_1b" not in segments

    def test_segment_cutter_different_angles(self, segment_assembly):
        """Segment cutters should work with different cut angles."""
        parts = [
            load_part(segment_assembly["outer_1"]),
            load_part(segment_assembly["inner_1a"]),
        ]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for name, shape, loc, rgb, seg in info:
            builder.Add(compound, apply_location(shape, loc))

        bbox = Bnd_Box()
        BRepBndLib.Add_s(compound, bbox)
        bbox_vals = bbox.Get()

        for angle in (90.0, 120.0, 180.0):
            cutter_a = make_segment_cutter(bbox_vals, angle, AXIS_MAP["z"], "a")
            for name, shape, loc, rgb, seg in info:
                if seg == "a":
                    moved = apply_location(shape, loc)
                    result = cut_assembly(moved, cutter_a)
                    assert not result.IsNull(), f"Cut failed for angle {angle}"


# ============================================================================
# Cylinder orientation (--cyl)
# ============================================================================

class TestOrientToCylinder:
    """Tests for orient_to_cylinder()."""

    # ------------------------------------------------------------------
    # Axis alignment
    # ------------------------------------------------------------------

    def test_cylinder_along_x_rotates_to_z(self):
        """A cylinder whose axis is along X should be reoriented so its axis is Z."""
        cyl_x = cq.Workplane("YZ").cylinder(50, 5)
        shape_before = cyl_x.val().wrapped
        xn, yn, zn, xx, yx, zx = get_bounding_box(shape_before)
        assert xx - xn > zx - zn, "Precondition: taller along X"

        rotated = orient_to_cylinder([(cyl_x, "outer_1")])
        assert len(rotated) == 1
        shape_after = rotated[0][0].val().wrapped
        xn2, yn2, zn2, xx2, yx2, zx2 = get_bounding_box(shape_after)
        assert zx2 - zn2 > xx2 - xn2, "After orient, Z span should exceed X span"

    def test_cylinder_along_y_rotates_to_z(self):
        """A cylinder whose axis is along Y should be reoriented so its axis is Z."""
        cyl_y = cq.Workplane("XZ").cylinder(50, 5)
        shape_before = cyl_y.val().wrapped
        xn, yn, zn, xx, yx, zx = get_bounding_box(shape_before)
        assert yx - yn > zx - zn, "Precondition: taller along Y"

        rotated = orient_to_cylinder([(cyl_y, "outer_1")])
        shape_after = rotated[0][0].val().wrapped
        xn2, yn2, zn2, xx2, yx2, zx2 = get_bounding_box(shape_after)
        assert zx2 - zn2 > yx2 - yn2, "After orient, Z span should exceed Y span"

    def test_cylinder_already_along_z_preserved(self):
        """A cylinder already along Z should keep the same height."""
        cyl_z = cq.Workplane("XY").cylinder(50, 5)
        xn, yn, zn, xx, yx, zx = get_bounding_box(cyl_z.val().wrapped)
        z_span_before = zx - zn

        rotated = orient_to_cylinder([(cyl_z, "outer_1")])
        shape_after = rotated[0][0].val().wrapped
        xn2, yn2, zn2, xx2, yx2, zx2 = get_bounding_box(shape_after)
        assert abs((zx2 - zn2) - z_span_before) < 1.0

    def test_multiple_parts_rotated_consistently(self):
        """All parts in a multi-part set should receive the same rotation."""
        outer = cq.Workplane("YZ").cylinder(50, 10)
        inner = cq.Workplane("YZ").cylinder(50, 5)

        rotated = orient_to_cylinder([(outer, "outer_1"), (inner, "inner_1")])
        assert len(rotated) == 2
        for wp, name in rotated:
            shape = wp.val().wrapped
            xn, yn, zn, xx, yx, zx = get_bounding_box(shape)
            assert zx - zn > xx - xn, f"{name}: Z span should exceed X span after orient"

    # ------------------------------------------------------------------
    # Wider end down (conic parts)
    # ------------------------------------------------------------------

    def _make_frustum_wide_at_top(self):
        """Return a frustum (truncated cone) along Z with the wide end at the top."""
        # Build wide-base-down first, then flip it
        pts = [(5, 0), (15, 0), (15, 30), (5, 30)]
        frustum = (
            cq.Workplane("XZ")
            .polyline(pts)
            .close()
            .revolve(360, (0, 0, 0), (0, 1, 0))
        )
        # Flip so wide end is at top
        flip = gp_Trsf()
        flip.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.pi)
        flipped = BRepBuilderAPI_Transform(frustum.val().wrapped, flip, True).Shape()
        return cq.Workplane("XY").newObject([cq.Shape(flipped)])

    def test_cone_wide_end_down_after_orient(self):
        """A cone/frustum with wide end at top should be flipped so wide end is down."""
        wp = self._make_frustum_wide_at_top()
        shape_before = wp.val().wrapped
        # Verify precondition
        xn, yn, zn, xx, yx, zx = get_bounding_box(shape_before)
        verts, _ = tessellate_shape(shape_before, tolerance=0.5)
        v = np.array(verts)
        z_mid = (zn + zx) / 2
        r_bot = np.mean(np.hypot(v[v[:, 2] < z_mid, 0], v[v[:, 2] < z_mid, 1]))
        r_top = np.mean(np.hypot(v[v[:, 2] >= z_mid, 0], v[v[:, 2] >= z_mid, 1]))
        assert r_top > r_bot, "Precondition: frustum has wider end at top"

        result = orient_to_cylinder([(wp, "outer_1")])
        res_shape = result[0][0].val().wrapped
        xn2, yn2, zn2, xx2, yx2, zx2 = get_bounding_box(res_shape)
        verts2, _ = tessellate_shape(res_shape, tolerance=0.5)
        v2 = np.array(verts2)
        z_mid2 = (zn2 + zx2) / 2
        r_bot2 = np.mean(np.hypot(v2[v2[:, 2] < z_mid2, 0], v2[v2[:, 2] < z_mid2, 1]))
        r_top2 = np.mean(np.hypot(v2[v2[:, 2] >= z_mid2, 0], v2[v2[:, 2] >= z_mid2, 1]))
        assert r_bot2 > r_top2, "After orient, wider end should be at bottom"

    def test_cone_already_wide_end_down_not_flipped(self):
        """A cone already wide-side-down should not be flipped."""
        pts = [(5, 0), (15, 0), (15, 30), (5, 30)]
        frustum = (
            cq.Workplane("XZ")
            .polyline(pts)
            .close()
            .revolve(360, (0, 0, 0), (0, 1, 0))
        )
        shape = frustum.val().wrapped
        xn, yn, zn, xx, yx, zx = get_bounding_box(shape)
        verts, _ = tessellate_shape(shape, tolerance=0.5)
        v = np.array(verts)
        z_mid = (zn + zx) / 2
        r_bot = np.mean(np.hypot(v[v[:, 2] < z_mid, 0], v[v[:, 2] < z_mid, 1]))
        r_top = np.mean(np.hypot(v[v[:, 2] >= z_mid, 0], v[v[:, 2] >= z_mid, 1]))
        assert r_bot > r_top, "Precondition: frustum already wide-side-down"

        result = orient_to_cylinder([(frustum, "outer_1")])
        res_shape = result[0][0].val().wrapped
        xn2, yn2, zn2, xx2, yx2, zx2 = get_bounding_box(res_shape)
        verts2, _ = tessellate_shape(res_shape, tolerance=0.5)
        v2 = np.array(verts2)
        z_mid2 = (zn2 + zx2) / 2
        r_bot2 = np.mean(np.hypot(v2[v2[:, 2] < z_mid2, 0], v2[v2[:, 2] < z_mid2, 1]))
        r_top2 = np.mean(np.hypot(v2[v2[:, 2] >= z_mid2, 0], v2[v2[:, 2] >= z_mid2, 1]))
        assert r_bot2 > r_top2, "Wide end should still be at bottom"

    # ------------------------------------------------------------------
    # Restacking
    # ------------------------------------------------------------------

    def test_restack_no_gap_contiguous(self):
        """After orient, parts should sit directly on top of each other (gap=0)."""
        # Three cylinders already along Z but offset with gaps between them
        c1 = cq.Workplane("XY").cylinder(20, 5)
        c2 = cq.Workplane("XY").cylinder(20, 5).translate((0, 0, 30))
        c3 = cq.Workplane("XY").cylinder(20, 5).translate((0, 0, 65))

        result = orient_to_cylinder(
            [(c1, "outer_1"), (c2, "outer_2"), (c3, "outer_3")], gap=0.0
        )
        assert len(result) == 3

        bboxes = sorted(
            [get_bounding_box(wp.val().wrapped) for wp, _ in result],
            key=lambda b: b[2],
        )

        tol = 0.1
        assert bboxes[0][2] == pytest.approx(0.0, abs=tol), "First part bottom at Z=0"
        assert bboxes[1][2] == pytest.approx(bboxes[0][5], abs=tol), \
            "Second part bottom should equal first part top"
        assert bboxes[2][2] == pytest.approx(bboxes[1][5], abs=tol), \
            "Third part bottom should equal second part top"

    def test_restack_with_gap(self):
        """After orient with gap>0, parts should have the specified gap between them."""
        gap = 5.0
        c1 = cq.Workplane("XY").cylinder(20, 5)
        c2 = cq.Workplane("XY").cylinder(20, 5).translate((0, 0, 50))

        result = orient_to_cylinder(
            [(c1, "outer_1"), (c2, "outer_2")], gap=gap
        )
        assert len(result) == 2

        bboxes = sorted(
            [get_bounding_box(wp.val().wrapped) for wp, _ in result],
            key=lambda b: b[2],
        )
        gap_actual = bboxes[1][2] - bboxes[0][5]
        assert gap_actual == pytest.approx(gap, abs=0.1), \
            f"Gap should be {gap}, got {gap_actual}"

    def test_restack_bottom_at_zero(self):
        """After orient, the bottommost part's minimum Z should be 0."""
        cyl = cq.Workplane("YZ").cylinder(30, 8).translate((50, 0, 0))
        result = orient_to_cylinder([(cyl, "outer_1")])
        xn, yn, zn, xx, yx, zx = get_bounding_box(result[0][0].val().wrapped)
        assert zn == pytest.approx(0.0, abs=0.1)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_parts_list_returns_empty(self):
        assert orient_to_cylinder([]) == []

    def test_names_preserved(self):
        cyl = cq.Workplane("YZ").cylinder(50, 5)
        result = orient_to_cylinder([(cyl, "outer_1")])
        assert result[0][1] == "outer_1"


# ============================================================================
# Test: Cut robustness (validity of results)
# ============================================================================

class TestCutRobustness:
    """Ensure cut_assembly always returns a valid shape for complex geometry."""

    def _get_bbox_vals(self, shape):
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        return bbox.Get()

    def test_cut_result_is_valid_simple_box(self):
        """Cut of a simple box should always produce a valid shape."""
        from OCP.BRepCheck import BRepCheck_Analyzer

        box = cq.Workplane("XY").box(20, 20, 20).val().wrapped
        bbox_vals = self._get_bbox_vals(box)

        for angle in [45, 90, 135, 180, 225, 270, 315]:
            cutter = make_cutter(bbox_vals, angle, AXIS_MAP["z"])
            result = cut_assembly(box, cutter)
            assert not result.IsNull(), f"angle={angle}: result is null"
            assert BRepCheck_Analyzer(result).IsValid(), (
                f"angle={angle}: result shape is topologically invalid"
            )

    def test_cut_result_is_valid_all_axes(self):
        """Cut is valid for all three stacking axes."""
        from OCP.BRepCheck import BRepCheck_Analyzer

        box = cq.Workplane("XY").box(15, 15, 30).val().wrapped
        bbox_vals = self._get_bbox_vals(box)

        for axis_name in ("x", "y", "z"):
            cutter = make_cutter(bbox_vals, 90, AXIS_MAP[axis_name])
            result = cut_assembly(box, cutter)
            assert not result.IsNull(), f"axis={axis_name}: result is null"
            assert BRepCheck_Analyzer(result).IsValid(), (
                f"axis={axis_name}: result shape is topologically invalid"
            )

    def test_cut_real_step_file(self, tmp_dir):
        """Cut of the bundled cut.step at several angles must produce valid shapes."""
        import os
        from OCP.BRepCheck import BRepCheck_Analyzer

        step_path = os.path.join(os.path.dirname(__file__), "cut.step")
        if not os.path.exists(step_path):
            pytest.skip("cut.step not found")

        wp, name = load_part(step_path)
        shape = wp.val().wrapped
        bbox_vals = self._get_bbox_vals(shape)

        for angle in [90, 180, 270]:
            cutter = make_cutter(bbox_vals, angle, AXIS_MAP["z"])
            result = cut_assembly(shape, cutter)
            assert not result.IsNull(), f"angle={angle}: result is null"
            assert BRepCheck_Analyzer(result).IsValid(), (
                f"angle={angle}: result shape invalid for cut.step"
            )

    def test_mesh_shell_to_solid_adaptive_tolerance(self, tmp_dir):
        """_mesh_shell_to_solid should succeed even on meshes with gaps."""
        from OCP.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND

        # Build a small box and export as STL, then reload as a shell
        box = cq.Workplane("XY").box(5, 5, 5)
        stl_path = os.path.join(tmp_dir, "box.stl")
        box.val().exportStl(stl_path)

        from OCP.StlAPI import StlAPI_Reader
        from OCP.TopoDS import TopoDS_Shape

        reader = StlAPI_Reader()
        raw = TopoDS_Shape()
        reader.Read(raw, stl_path)

        solid = _mesh_shell_to_solid(raw)
        # Should be promoted to SOLID (or at least not null)
        assert not solid.IsNull()
        assert solid.ShapeType() == TopAbs_SOLID


# ============================================================================
# Test: Direct geometry cutting
# ============================================================================

class TestDirectGeometryCut:
    """Tests for cut_part_direct() — the split-and-filter approach."""

    def _get_volume(self, shape):
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        return props.Mass()

    def _origin_for(self, shape, axis_vec):
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        bbox_vals = bbox.Get()
        _, _, origin_pt, _ = _cutter_params(bbox_vals, axis_vec)
        return origin_pt

    def test_direct_cut_90_cylinder(self):
        """90° direct cut of a cylinder removes exactly one quarter."""
        cyl = cq.Workplane("XY").cylinder(30, 8).val().wrapped
        orig_vol = self._get_volume(cyl)
        origin = self._origin_for(cyl, AXIS_MAP["z"])

        result = cut_part_direct(cyl, 90, AXIS_MAP["z"], origin)
        assert result is not None
        result_vol = self._get_volume(result)
        assert abs(result_vol / orig_vol - 0.75) < 0.02, (
            f"Expected ~75% kept, got {result_vol/orig_vol:.4f}"
        )

    def test_direct_cut_180_cylinder(self):
        """180° direct cut of a cylinder removes exactly one half."""
        cyl = cq.Workplane("XY").cylinder(30, 8).val().wrapped
        orig_vol = self._get_volume(cyl)
        origin = self._origin_for(cyl, AXIS_MAP["z"])

        result = cut_part_direct(cyl, 180, AXIS_MAP["z"], origin)
        assert result is not None
        result_vol = self._get_volume(result)
        assert abs(result_vol / orig_vol - 0.50) < 0.02, (
            f"Expected ~50% kept, got {result_vol/orig_vol:.4f}"
        )

    def test_direct_cut_90_box(self):
        """90° direct cut of a box removes approximately one quarter."""
        box = cq.Workplane("XY").box(20, 20, 10).val().wrapped
        orig_vol = self._get_volume(box)
        origin = self._origin_for(box, AXIS_MAP["z"])

        result = cut_part_direct(box, 90, AXIS_MAP["z"], origin)
        assert result is not None
        result_vol = self._get_volume(result)
        assert abs(result_vol / orig_vol - 0.75) < 0.02

    def test_direct_cut_flange(self):
        """Direct cut of a flange (cylinder with hex cutout)."""
        flange = (
            cq.Workplane("XY")
            .circle(15).extrude(5)
            .faces(">Z").workplane()
            .polygon(6, 12).cutBlind(-5)
        ).val().wrapped
        orig_vol = self._get_volume(flange)
        origin = self._origin_for(flange, AXIS_MAP["z"])

        result = cut_part_direct(flange, 90, AXIS_MAP["z"], origin)
        assert result is not None
        result_vol = self._get_volume(result)
        assert abs(result_vol / orig_vol - 0.75) < 0.05

    def test_direct_cut_all_axes(self):
        """Direct cut works for all three stacking axes."""
        box = cq.Workplane("XY").box(15, 15, 30).val().wrapped

        for axis_name in ("x", "y", "z"):
            axis_vec = AXIS_MAP[axis_name]
            origin = self._origin_for(box, axis_vec)
            result = cut_part_direct(box, 90, axis_vec, origin)
            assert result is not None, f"axis={axis_name}: result is None"
            bbox = Bnd_Box()
            BRepBndLib.Add_s(result, bbox)
            assert not bbox.IsVoid(), f"axis={axis_name}: result bbox is void"

    def test_direct_cut_multiple_angles(self):
        """Direct cut produces valid results for a range of angles."""
        cyl = cq.Workplane("XY").cylinder(20, 10).val().wrapped
        orig_vol = self._get_volume(cyl)
        origin = self._origin_for(cyl, AXIS_MAP["z"])

        for angle in [45, 90, 120, 180, 270]:
            result = cut_part_direct(cyl, angle, AXIS_MAP["z"], origin)
            assert result is not None, f"angle={angle}: result is None"
            result_vol = self._get_volume(result)
            expected_ratio = (360 - angle) / 360.0
            assert abs(result_vol / orig_vol - expected_ratio) < 0.03, (
                f"angle={angle}: expected {expected_ratio:.3f}, "
                f"got {result_vol/orig_vol:.3f}"
            )

    def test_direct_cut_produces_cap_faces(self):
        """Direct cut should produce planar cap faces at the cut boundaries."""
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopoDS import TopoDS
        from OCP.BRep import BRep_Tool
        from OCP.GeomAdaptor import GeomAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Plane

        cyl = cq.Workplane("XY").cylinder(30, 8).val().wrapped
        origin = self._origin_for(cyl, AXIS_MAP["z"])

        result = cut_part_direct(cyl, 90, AXIS_MAP["z"], origin)
        assert result is not None

        # Count planar faces — original cylinder has 2 (top+bottom),
        # direct cut should add 2 more (cap at 0° and cap at 90°)
        plane_count = 0
        explorer = TopExp_Explorer(result, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            surface = BRep_Tool.Surface_s(face)
            adaptor = GeomAdaptor_Surface(surface)
            if adaptor.GetType() == GeomAbs_Plane:
                plane_count += 1
            explorer.Next()

        # Original has 2 plane faces; after 90° cut the splitter creates
        # additional planar caps. Expect at least 4 planar faces.
        assert plane_count >= 4, (
            f"Expected at least 4 planar faces (2 original + 2 caps), "
            f"got {plane_count}"
        )

    def test_direct_cut_result_exportable(self, tmp_dir):
        """Direct cut result can be exported as STEP."""
        box = cq.Workplane("XY").box(20, 20, 10).val().wrapped
        origin = self._origin_for(box, AXIS_MAP["z"])

        result = cut_part_direct(box, 90, AXIS_MAP["z"], origin)
        assert result is not None

        out_path = os.path.join(tmp_dir, "direct_cut.step")
        export_shape_step(result, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 100

    def test_direct_segment_a_and_b(self):
        """Direct segment cut splits the remaining arc into two halves."""
        cyl = cq.Workplane("XY").cylinder(30, 8).val().wrapped
        orig_vol = self._get_volume(cyl)
        origin = self._origin_for(cyl, AXIS_MAP["z"])
        cut_angle = 90.0

        seg_a = cut_part_direct_segment(
            cyl, cut_angle, AXIS_MAP["z"], origin, "a"
        )
        seg_b = cut_part_direct_segment(
            cyl, cut_angle, AXIS_MAP["z"], origin, "b"
        )

        assert seg_a is not None
        assert seg_b is not None

        vol_a = self._get_volume(seg_a)
        vol_b = self._get_volume(seg_b)

        # Each segment should be roughly half of the remaining arc (270°/2 = 135°)
        expected_each = orig_vol * 135.0 / 360.0
        assert abs(vol_a - expected_each) / expected_each < 0.05, (
            f"Segment a: expected ~{expected_each:.1f}, got {vol_a:.1f}"
        )
        assert abs(vol_b - expected_each) / expected_each < 0.05, (
            f"Segment b: expected ~{expected_each:.1f}, got {vol_b:.1f}"
        )

    def test_direct_cut_real_step_file(self):
        """Direct cut of the bundled cut.step at several angles."""
        step_path = os.path.join(os.path.dirname(__file__), "cut.step")
        if not os.path.exists(step_path):
            pytest.skip("cut.step not found")

        wp, name = load_part(step_path)
        shape = wp.val().wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])

        for angle in [90, 180]:
            result = cut_part_direct(shape, angle, AXIS_MAP["z"], origin)
            assert result is not None, f"angle={angle}: result is None"
            result_bbox = Bnd_Box()
            BRepBndLib.Add_s(result, result_bbox)
            assert not result_bbox.IsVoid(), (
                f"angle={angle}: result bbox is void"
            )


# ============================================================================
# Test: Outer stackup gap — outer_1.STEP and outer_2.STEP real files
# ============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUTER1_PATH = os.path.join(_HERE, "outer_1.STEP")
_OUTER2_PATH = os.path.join(_HERE, "outer_2.STEP")
_OUTER_FILES_PRESENT = os.path.exists(_OUTER1_PATH) and os.path.exists(_OUTER2_PATH)


@pytest.mark.skipif(
    not _OUTER_FILES_PRESENT,
    reason="outer_1.STEP / outer_2.STEP not present in repo",
)
class TestOuterStackupGap:
    """Regression tests for the large-gap bug with outer_1.STEP / outer_2.STEP.

    These parts are cylindrical tubes oriented with their long axis along Y in
    the original STEP files.  Without --cyl (orient_to_cylinder), stack_parts
    uses the Z bounding-box extent (the tube *diameter*, ~511 mm and ~440 mm)
    as the stacking height rather than the actual tube length (~917 mm and
    ~499 mm).  This produces a large visual gap: outer_1 appears ~917 mm wide
    in the Y direction while outer_2 is only ~498 mm wide — a ~209 mm overhang
    gap at each end.

    With orient_to_cylinder the parts are rotated so the long axis aligns with
    Z, the correct heights are used, and the Y-direction overhang shrinks to
    ~35 mm (just the radial difference between the two tube diameters).
    """

    def _load_and_stack(self, use_cyl=False, gap=0.0):
        """Load outer_1/outer_2 STEP files, optionally orient, then stack."""
        parts = [load_part(_OUTER1_PATH), load_part(_OUTER2_PATH)]
        if use_cyl:
            parts = orient_to_cylinder(parts, gap=gap)
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=gap)
        return [
            (name, apply_location(shape, loc))
            for name, shape, loc, rgb, seg in info
        ]

    # ------------------------------------------------------------------
    # Tests that expose the bug (no --cyl / orient_to_cylinder)
    # ------------------------------------------------------------------

    def test_without_cyl_outer1_height_is_diameter_not_length(self):
        """Without --cyl, outer_1's stacked Z-height equals its tube diameter
        (~511 mm), NOT its actual tube length (~495 mm).  This is the root
        cause of the visual stackup issue."""
        moved = self._load_and_stack(use_cyl=False)
        outer1 = next(s for n, s in moved if n == "outer_1")
        _, _, zn, _, _, zx = get_tight_bounding_box(outer1)
        height = zx - zn
        # Should be ~511 mm (diameter), not ~495 mm (tube length)
        assert abs(height - 510.7) < 5.0, (
            f"Expected ~510.7 mm (tube diameter), got {height:.1f} mm"
        )

    def test_without_cyl_large_y_direction_gap(self):
        """Without --cyl, the tube long-axes lie along Y.  Both tubes have
        similar Y extents (their actual tube lengths, ~495–500 mm each).
        The visual problem without --cyl is that Z-stacking uses the tube
        *diameter* (~511 mm / ~440 mm) instead of the tube length, so the
        assembly looks wrong in 3-D even though the Z bboxes touch.

        We verify the tight Y extents are the actual tube lengths (both
        ≈495–502 mm) — confirming the degenerate-face-inflated value
        (~917 mm) is NOT used for positioning."""
        moved = self._load_and_stack(use_cyl=False)
        outer1 = next(s for n, s in moved if n == "outer_1")
        outer2 = next(s for n, s in moved if n == "outer_2")

        # After stack_parts runs (which internally calls get_tight_bounding_box),
        # BRepMesh side-effects mean get_bounding_box also returns tight bbox.
        _, y1n, _, _, y1x, _ = get_tight_bounding_box(outer1)
        _, y2n, _, _, y2x, _ = get_tight_bounding_box(outer2)
        y_extent_1 = y1x - y1n  # actual tube length of outer_1, ~495 mm
        y_extent_2 = y2x - y2n  # actual tube length of outer_2, ~499 mm

        # Neither tube's Y extent should be inflated to ~917 mm.
        assert y_extent_1 < 560.0, (
            f"outer_1 tight Y extent should be tube length (<560 mm), "
            f"got {y_extent_1:.1f} mm — degenerate face may be inflating bbox"
        )
        assert y_extent_2 < 560.0, (
            f"outer_2 tight Y extent should be tube length (<560 mm), "
            f"got {y_extent_2:.1f} mm"
        )
        # Both tubes have similar lengths (within 10 mm of each other).
        assert abs(y_extent_1 - y_extent_2) < 10.0, (
            f"Tube Y extents should be similar (both ~495–500 mm), "
            f"got outer_1={y_extent_1:.1f} mm, outer_2={y_extent_2:.1f} mm"
        )

    def test_without_cyl_z_bboxes_touch(self):
        """Without --cyl the Z bounding boxes still touch (gap=0 in Z).
        The visual problem is the wrong stacking height (diameter not length),
        not a Z gap."""
        moved = self._load_and_stack(use_cyl=False)
        outer1 = next(s for n, s in moved if n == "outer_1")
        outer2 = next(s for n, s in moved if n == "outer_2")
        _, _, _, _, _, z1x = get_tight_bounding_box(outer1)
        _, _, z2n, _, _, _  = get_tight_bounding_box(outer2)
        assert abs(z2n - z1x) < 1.0, (
            f"Z gap without --cyl: {z2n - z1x:.3f} mm (expected 0)"
        )

    # ------------------------------------------------------------------
    # Tests that verify the fix (with --cyl / orient_to_cylinder)
    # ------------------------------------------------------------------

    def test_with_cyl_outer1_height_is_tube_length(self):
        """With --cyl, outer_1's stacked Z-height equals its actual tube
        body height (~498.5 mm), not the degenerate-face-inflated value
        (~917 mm) nor the tube diameter."""
        moved = self._load_and_stack(use_cyl=True)
        outer1 = next(s for n, s in moved if n == "outer_1")
        xn, yn, zn, xx, yx, zx = get_tight_bounding_box(outer1)
        height = zx - zn
        assert abs(height - 498.5) < 5.0, (
            f"Expected ~498.5 mm (tube body height), got {height:.1f} mm"
        )

    def test_with_cyl_outer2_height_is_tube_length(self):
        """With --cyl, outer_2's stacked Z-height equals its actual tube
        length (~499 mm), not its diameter."""
        moved = self._load_and_stack(use_cyl=True)
        outer2 = next(s for n, s in moved if n == "outer_2")
        _, _, zn, _, _, zx = get_tight_bounding_box(outer2)
        height = zx - zn
        assert abs(height - 499.0) < 5.0, (
            f"Expected ~499.0 mm (tube length), got {height:.1f} mm"
        )

    def test_with_cyl_no_z_gap(self):
        """With --cyl, outer_2 starts exactly where outer_1 ends (no Z gap)."""
        moved = self._load_and_stack(use_cyl=True)
        outer1 = next(s for n, s in moved if n == "outer_1")
        outer2 = next(s for n, s in moved if n == "outer_2")
        _, _, _, _, _, z1x = get_tight_bounding_box(outer1)
        _, _, z2n, _, _, _  = get_tight_bounding_box(outer2)
        assert abs(z2n - z1x) < 1.0, (
            f"Z gap with --cyl: {z2n - z1x:.3f} mm (expected 0)"
        )

    def test_with_cyl_outer1_base_at_z0(self):
        """With --cyl, outer_1's base is placed at z=0."""
        moved = self._load_and_stack(use_cyl=True)
        outer1 = next(s for n, s in moved if n == "outer_1")
        _, _, zn, _, _, _ = get_tight_bounding_box(outer1)
        assert abs(zn) < 1.0, f"outer_1 base at z={zn:.3f}, expected ~0"

    def test_with_cyl_small_y_direction_gap(self):
        """With --cyl, the Y extents are the tube diameters (~511 mm and
        ~440 mm).  The Y-direction overhang gap shrinks to ~35 mm per end
        (just the radial difference), confirming the large gap is fixed."""
        moved = self._load_and_stack(use_cyl=True)
        outer1 = next(s for n, s in moved if n == "outer_1")
        outer2 = next(s for n, s in moved if n == "outer_2")

        _, y1n, _, _, y1x, _ = get_tight_bounding_box(outer1)
        _, y2n, _, _, y2x, _ = get_tight_bounding_box(outer2)
        y_extent_1 = y1x - y1n  # ~511 mm (outer_1 diameter)
        y_extent_2 = y2x - y2n  # ~440 mm (outer_2 diameter)

        assert abs(y_extent_1 - 510.8) < 10.0, (
            f"outer_1 Y extent expected ~510.8 mm (diameter), "
            f"got {y_extent_1:.1f} mm"
        )
        assert abs(y_extent_2 - 439.9) < 10.0, (
            f"outer_2 Y extent expected ~439.9 mm (diameter), "
            f"got {y_extent_2:.1f} mm"
        )

        gap_each_end = (y_extent_1 - y_extent_2) / 2.0
        assert gap_each_end < 50.0, (
            f"Y-direction overhang gap with --cyl should be <50 mm, "
            f"got {gap_each_end:.1f} mm"
        )

    def test_with_cyl_both_xy_centered(self):
        """With --cyl, both parts are XY-centered at the origin.

        Use the exact (analytical) bbox rather than the tight/triangulation bbox
        here because apply_location returns a shape with a TopLoc_Location, and
        BRepBndLib with useTriangulation=True does not reliably apply that
        location to the mesh vertices in all OCC versions.  The exact bbox is
        always location-aware.  After the --cyl rotation, the degenerate BSpline
        face inflates only the Z bbox (tube axis), not X or Y, so the exact XY
        extents are correct for this check."""
        moved = self._load_and_stack(use_cyl=True)
        for name, shape in moved:
            xn, yn, _, xx, yx, _ = get_bounding_box(shape)
            cx = (xn + xx) / 2.0
            cy = (yn + yx) / 2.0
            assert abs(cx) < 1.0, f"{name} not centered in X: cx={cx:.3f}"
            assert abs(cy) < 1.0, f"{name} not centered in Y: cy={cy:.3f}"


# ============================================================================
# Test: Physics simulation
# ============================================================================

from assemble import (
    simulate_physics,
    _shape_to_clean_trimesh,
    _find_support_drop,
    _find_support_drop_raycast,
    _find_support_drop_collision,
    _report_nesting,
)


class TestShapeToCleanTrimesh:
    """Tests for OCP shape → cleaned trimesh conversion."""

    def test_box_produces_valid_mesh(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        mesh = _shape_to_clean_trimesh(box.val().wrapped)
        assert mesh is not None
        assert len(mesh.faces) > 0
        assert len(mesh.vertices) > 0

    def test_degenerate_removed(self):
        """A valid shape should yield a mesh with no degenerate faces."""
        cyl = cq.Workplane("XY").cylinder(20, 5)
        mesh = _shape_to_clean_trimesh(cyl.val().wrapped)
        assert mesh is not None
        # All faces should have non-zero area
        areas = mesh.area_faces
        assert all(a > 0 for a in areas)

    def test_empty_shape_returns_none(self):
        """An empty compound should return None."""
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        mesh = _shape_to_clean_trimesh(compound)
        assert mesh is None


class TestPhysicsSimulation:
    """Tests for the gravity-settle physics simulation."""

    def test_single_part_settles_on_floor(self):
        """A single box should settle at z=0 (the floor)."""
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        # Start the box elevated at z=20
        loc = Location(Vector(0, 0, 20))
        part_info = [("box", shape, loc, (0.5, 0.5, 0.5), None)]

        result = simulate_physics(
            part_info, Vector(0, 0, 1), gap=0.0,
            max_iters=50,
        )

        assert len(result) == 1
        moved = apply_location(result[0][1], result[0][2])
        bb = get_bounding_box(moved)
        # z_min should be near 0 (floor)
        assert abs(bb[2]) < 1.0, f"Box z_min should be near 0, got {bb[2]}"

    def test_two_stacked_boxes(self):
        """Two boxes stacked with a gap should settle together."""
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped

        part_info = [
            ("box_1", shape, Location(Vector(0, 0, 0)), (0.5, 0.5, 0.5), None),
            ("box_2", shape, Location(Vector(0, 0, 15)), (0.7, 0.3, 0.3), None),
        ]

        result = simulate_physics(
            part_info, Vector(0, 0, 1), gap=5.0,
            max_iters=50,
        )

        assert len(result) == 2
        # Both boxes should be touching or nearly touching
        moved_1 = apply_location(result[0][1], result[0][2])
        moved_2 = apply_location(result[1][1], result[1][2])
        bb1 = get_bounding_box(moved_1)
        bb2 = get_bounding_box(moved_2)

        # Sort by z_min
        bboxes = sorted([(bb1, "box_1"), (bb2, "box_2")], key=lambda x: x[0][2])
        lower_zmax = bboxes[0][0][5]
        upper_zmin = bboxes[1][0][2]
        # Gap between them should be small (settled)
        gap_val = upper_zmin - lower_zmax
        assert gap_val < 2.0, f"Gap between stacked boxes should be small, got {gap_val}"

    def test_nesting_reduces_gap(self):
        """Concentric cylinders (hollow outer, solid inner) should nest,
        producing a negative bounding-box gap."""
        # Outer: hollow cylinder (shell)
        outer = cq.Workplane("XY").cylinder(20, 15).cut(
            cq.Workplane("XY").cylinder(20, 12)
        )
        # Inner: solid cylinder that fits inside
        inner = cq.Workplane("XY").cylinder(18, 11)

        outer_shape = outer.val().wrapped
        inner_shape = inner.val().wrapped

        # Stack inner on top (above outer)
        part_info = [
            ("outer_1", outer_shape, Location(Vector(0, 0, 0)),
             (0.5, 0.5, 0.5), None),
            ("inner_1", inner_shape, Location(Vector(0, 0, 25)),
             (0.8, 0.3, 0.3), None),
        ]

        result = simulate_physics(
            part_info, Vector(0, 0, 1), gap=0.0,
            max_iters=50,
        )

        # After settling, the inner cylinder should have dropped inside
        moved_outer = apply_location(result[0][1], result[0][2])
        moved_inner = apply_location(result[1][1], result[1][2])
        bb_outer = get_bounding_box(moved_outer)
        bb_inner = get_bounding_box(moved_inner)

        # The inner part's z_min should be at or below the outer part's z_max
        # (negative gap means nesting)
        outer_zmax = bb_outer[5]
        inner_zmin = bb_inner[2]
        effective_gap = inner_zmin - outer_zmax
        assert effective_gap < 1.0, (
            f"Inner should nest inside outer (gap < 1.0), got {effective_gap:.4f}"
        )

    def test_preserves_part_metadata(self):
        """Simulation should preserve name, color, and segment info."""
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        part_info = [
            ("test_part", shape, Location(Vector(0, 0, 0)),
             (0.1, 0.2, 0.3), "a"),
        ]

        result = simulate_physics(
            part_info, Vector(0, 0, 1), gap=0.0,
        )

        assert result[0][0] == "test_part"
        assert result[0][3] == (0.1, 0.2, 0.3)
        assert result[0][4] == "a"

    def test_empty_input(self):
        """Empty input should return empty output."""
        result = simulate_physics([], Vector(0, 0, 1), gap=0.0)
        assert result == []

    def test_x_axis_simulation(self):
        """Physics should work along X axis too."""
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        loc = Location(Vector(30, 0, 0))
        part_info = [("box", shape, loc, (0.5, 0.5, 0.5), None)]

        result = simulate_physics(
            part_info, Vector(1, 0, 0), gap=0.0,
            max_iters=50,
        )

        moved = apply_location(result[0][1], result[0][2])
        bb = get_bounding_box(moved)
        # x_min should settle near 0 (floor along X)
        assert abs(bb[0]) < 2.0, f"Box x_min should be near 0, got {bb[0]}"


class TestTubeNestingCollision:
    """Regression tests: a smaller-diameter tube must not fall through a
    larger-diameter tube.  The collision-based binary search in
    ``_find_support_drop`` detects radial wall-to-wall contact that
    vertical ray-casting misses.
    """

    def _make_tube(self, height, outer_r, inner_r):
        """Create a hollow tube (open ends) centered at the origin."""
        tube = cq.Workplane("XY").cylinder(height, outer_r).cut(
            cq.Workplane("XY").cylinder(height, inner_r)
        )
        return tube.val().wrapped

    def test_smaller_tube_does_not_fall_through_larger(self):
        """A tube with OD < the outer tube's ID should be caught by the
        collision check when the wall thicknesses overlap radially.

        Setup: outer tube (h=100, OD=60, ID=50) sitting on the floor.
               inner tube (h=80, OD=52, ID=44) placed above.

        The inner tube's outer wall (r=26) exceeds the outer tube's inner
        wall (r=25) by 1 mm, so it can only nest ~partially before the
        walls collide.  Without the collision fix, the ray grid misses
        the wall entirely and the inner tube falls to the floor.
        """
        outer_shape = self._make_tube(100, 30, 25)  # OD=60, ID=50
        inner_shape = self._make_tube(80, 26, 22)   # OD=52, ID=44

        part_info = [
            ("outer_1", outer_shape, Location(Vector(0, 0, 0)),
             (0.5, 0.5, 0.5), None),
            ("inner_1", inner_shape, Location(Vector(0, 0, 100)),
             (0.8, 0.3, 0.3), None),
        ]

        result = simulate_physics(
            part_info, Vector(0, 0, 1), gap=0.0,
        )

        moved_outer = apply_location(result[0][1], result[0][2])
        moved_inner = apply_location(result[1][1], result[1][2])
        bb_outer = get_bounding_box(moved_outer)
        bb_inner = get_bounding_box(moved_inner)

        outer_zmax = bb_outer[5]
        inner_zmin = bb_inner[2]

        # The inner tube should NOT have fallen to the floor (z_min ≈ 0).
        # It should rest with its bottom near the outer tube's top, with
        # some nesting overlap but NOT full pass-through.
        assert inner_zmin > 10.0, (
            f"Inner tube fell through outer tube: inner z_min={inner_zmin:.1f} "
            f"(expected > 10, i.e. NOT at floor level)"
        )

    def test_collision_drop_detects_wall_contact(self):
        """Directly test _find_support_drop_collision on concentric tubes."""
        import trimesh

        outer_shape = self._make_tube(100, 30, 25)
        inner_shape = self._make_tube(80, 26, 22)

        outer_moved = apply_location(outer_shape, Location(Vector(0, 0, 0)))
        inner_moved = apply_location(inner_shape, Location(Vector(0, 0, 100)))

        outer_mesh = _shape_to_clean_trimesh(outer_moved)
        inner_mesh = _shape_to_clean_trimesh(inner_moved)

        assert outer_mesh is not None
        assert inner_mesh is not None

        drop = _find_support_drop_collision(inner_mesh, outer_mesh, ax=2)

        # Should detect a collision limit before the inner tube reaches the
        # floor (drop < 100).
        assert drop is not None, (
            "Collision detection returned None — wall contact not detected"
        )
        assert drop < 95.0, (
            f"Collision drop={drop:.1f}, expected < 95 (should stop before "
            f"inner tube exits outer tube)"
        )


class TestPhysPipeline:
    """Integration test: --phys flag in run_pipeline."""

    def test_phys_flag_accepted(self, concentric_parts, tmp_dir):
        """Verify --phys flag is accepted and doesn't error."""
        from assemble import run_pipeline
        import types

        args = types.SimpleNamespace(
            inputs=[concentric_parts["outer_1"], concentric_parts["outer_2"]],
            output=os.path.join(tmp_dir, "phys_test.step"),
            axis="z",
            gap=5.0,
            cut_angle=None,
            render=None,
            resolution=512,
            cyl=False,
            cut_direct=False,
            phys=True,
            debug=False,
        )
        ret = run_pipeline(args)
        assert ret == 0
        assert os.path.exists(args.output)


# ============================================================================
# Autoscale
# ============================================================================

class TestParseTargetDiameter:
    """Tests for _parse_target_diameter helper."""

    def test_basic(self):
        assert _parse_target_diameter("inner_2_3_d8") == 8.0

    def test_decimal(self):
        assert _parse_target_diameter("inner_1_d12.5") == 12.5

    def test_no_match(self):
        assert _parse_target_diameter("outer_1") is None

    def test_case_insensitive(self):
        assert _parse_target_diameter("inner_1_D10") == 10.0


class TestGetXYDiameter:
    """Tests for _get_xy_diameter helper."""

    def test_box(self):
        box = cq.Workplane("XY").box(20, 30, 10)
        diam = _get_xy_diameter(box.val().wrapped)
        assert abs(diam - 30.0) < 0.5  # max(20, 30) = 30

    def test_cylinder(self):
        cyl = cq.Workplane("XY").cylinder(10, 5)  # height=10, radius=5
        diam = _get_xy_diameter(cyl.val().wrapped)
        assert abs(diam - 10.0) < 0.5  # diameter = 2*radius = 10


class TestScaleShape:
    """Tests for scale_shape helper."""

    def test_double(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        scaled = scale_shape(shape, 2.0)
        xn, yn, zn, xx, yx, zx = get_tight_bounding_box(scaled)
        assert abs((xx - xn) - 20.0) < 0.5
        assert abs((yx - yn) - 20.0) < 0.5
        assert abs((zx - zn) - 20.0) < 0.5

    def test_half(self):
        box = cq.Workplane("XY").box(10, 10, 10)
        shape = box.val().wrapped
        scaled = scale_shape(shape, 0.5)
        xn, yn, zn, xx, yx, zx = get_tight_bounding_box(scaled)
        assert abs((xx - xn) - 5.0) < 0.5


class TestAutoscaleParts:
    """Tests for autoscale_parts — unit mismatch detection and _dN scaling."""

    def test_no_scaling_when_similar_size(self):
        """Parts of similar size should not be rescaled."""
        outer = cq.Workplane("XY").cylinder(10, 25)  # diam ~50
        inner = cq.Workplane("XY").cylinder(10, 10)  # diam ~20
        parts = [(outer, "outer_1"), (inner, "inner_1")]
        result = autoscale_parts(parts)
        # inner is smaller but not drastically; no scaling expected
        orig_diam = _get_xy_diameter(inner.val().wrapped)
        result_diam = _get_xy_diameter(result[1][0].val().wrapped)
        assert abs(result_diam - orig_diam) < 0.5

    def test_mm_to_inch_detection(self):
        """Inner modelled in mm should be scaled up by ~25.4 when outer is in inches."""
        # Outer is 2 inches diameter = 50.8 mm, but in inches it's diam=2
        outer = cq.Workplane("XY").cylinder(10, 25)   # diam=50 (inch units)
        # Inner is 1.5 inches = 38.1 mm, but modelled in mm so diam=1.5
        inner = cq.Workplane("XY").cylinder(10, 0.75)  # diam=1.5 (mm value)
        parts = [(outer, "outer_1"), (inner, "inner_1")]
        result = autoscale_parts(parts)
        result_diam = _get_xy_diameter(result[1][0].val().wrapped)
        # 1.5 * 25.4 = 38.1, which is closer to 50 than 1.5 * 2.54 = 3.81
        assert result_diam > 30.0, f"Expected scaled up diameter, got {result_diam}"

    def test_d_tag_scaling(self):
        """Parts with _dN tag should be scaled to diameter N."""
        outer = cq.Workplane("XY").cylinder(10, 25)  # diam=50
        inner = cq.Workplane("XY").cylinder(10, 5)   # diam=10
        parts = [(outer, "outer_1"), (inner, "inner_1_d30")]
        result = autoscale_parts(parts)
        result_diam = _get_xy_diameter(result[1][0].val().wrapped)
        assert abs(result_diam - 30.0) < 1.0, f"Expected ~30, got {result_diam}"

    def test_outer_not_scaled(self):
        """Outer parts should never be scaled (they are the reference)."""
        outer = cq.Workplane("XY").cylinder(10, 25)  # diam=50
        inner = cq.Workplane("XY").cylinder(10, 0.75)
        parts = [(outer, "outer_1"), (inner, "inner_1")]
        result = autoscale_parts(parts)
        orig_diam = _get_xy_diameter(outer.val().wrapped)
        result_diam = _get_xy_diameter(result[0][0].val().wrapped)
        assert abs(result_diam - orig_diam) < 0.1

    def test_multi_level(self):
        """Each level's inner should be scaled independently."""
        outer1 = cq.Workplane("XY").cylinder(10, 25)   # diam=50
        inner1 = cq.Workplane("XY").cylinder(10, 0.75)  # tiny → mm heuristic
        outer2 = cq.Workplane("XY").cylinder(10, 25)
        inner2 = cq.Workplane("XY").cylinder(10, 10)   # normal → no scaling
        parts = [
            (outer1, "outer_1"), (inner1, "inner_1"),
            (outer2, "outer_2"), (inner2, "inner_2"),
        ]
        result = autoscale_parts(parts)
        # inner_1 should have been scaled
        d1 = _get_xy_diameter(result[1][0].val().wrapped)
        assert d1 > 10.0, f"inner_1 should be scaled up, got {d1}"
        # inner_2 should NOT have been scaled
        d2_orig = _get_xy_diameter(inner2.val().wrapped)
        d2_result = _get_xy_diameter(result[3][0].val().wrapped)
        assert abs(d2_result - d2_orig) < 0.5


# ============================================================================
# Test: Complex mesh cutting
# ============================================================================

class TestComplexMeshCutting:
    """Cut complex, real-world-like meshes and validate clean results.

    Each test builds a non-trivial geometry (thin walls, internal cavities,
    curved intersections, multi-body assemblies) and verifies that both
    boolean and direct cutting produce valid, watertight results with the
    correct volume ratio.
    """

    def _get_volume(self, shape):
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        return props.Mass()

    def _get_bbox_vals(self, shape):
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        return bbox.Get()

    def _origin_for(self, shape, axis_vec):
        bbox_vals = self._get_bbox_vals(shape)
        _, _, origin_pt, _ = _cutter_params(bbox_vals, axis_vec)
        return origin_pt

    def _assert_valid_cut(self, original, result, angle, tol=0.05, label=""):
        """Assert the cut result is non-null, has correct volume, and valid topology."""
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE

        assert result is not None, f"{label}: cut returned None"
        assert not result.IsNull(), f"{label}: cut result is null"

        result_bbox = Bnd_Box()
        BRepBndLib.Add_s(result, result_bbox)
        assert not result_bbox.IsVoid(), f"{label}: result bbox is void"

        # Volume check
        orig_vol = self._get_volume(original)
        cut_vol = self._get_volume(result)
        expected_ratio = (360.0 - angle) / 360.0
        actual_ratio = cut_vol / orig_vol if orig_vol > 1e-9 else 0
        assert abs(actual_ratio - expected_ratio) < tol, (
            f"{label}: volume ratio {actual_ratio:.4f} vs expected "
            f"{expected_ratio:.4f} (tol={tol})"
        )

        # Must have faces (tessellatable)
        exp = TopExp_Explorer(result, TopAbs_FACE)
        assert exp.More(), f"{label}: result has no faces"

        # Tessellation check
        verts, faces = tessellate_shape(result, tolerance=0.1)
        assert len(verts) > 0 and len(faces) > 0, (
            f"{label}: tessellation produced no geometry"
        )

    # ── Thin-wall hollow cylinder (pipe) ────────────────────────────

    def test_boolean_cut_thin_wall_pipe(self):
        """Boolean cut through a thin-walled pipe (2mm wall, 40mm OD)."""
        pipe = (cq.Workplane("XY")
                .cylinder(60, 20)
                .faces(">Z").workplane()
                .hole(36))  # 2mm wall
        shape = pipe.val().wrapped
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 90, tol=0.08,
                               label="thin-wall pipe boolean 90°")

    def test_direct_cut_thin_wall_pipe(self):
        """Direct cut through a thin-walled pipe."""
        pipe = (cq.Workplane("XY")
                .cylinder(60, 20)
                .faces(">Z").workplane()
                .hole(36))
        shape = pipe.val().wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 90, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 90, tol=0.08,
                               label="thin-wall pipe direct 90°")

    # ── Sphere with through-hole ────────────────────────────────────

    def test_boolean_cut_drilled_sphere(self):
        """Boolean cut on a sphere with a cylindrical bore through it."""
        sphere = cq.Workplane("XY").sphere(25)
        bore = cq.Workplane("XY").cylinder(60, 5)
        drilled = sphere.cut(bore)
        shape = drilled.val().wrapped
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 90, tol=0.10,
                               label="drilled sphere boolean 90°")

    def test_direct_cut_drilled_sphere(self):
        """Direct cut on a sphere with a bore."""
        sphere = cq.Workplane("XY").sphere(25)
        bore = cq.Workplane("XY").cylinder(60, 5)
        drilled = sphere.cut(bore)
        shape = drilled.val().wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 120, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 120, tol=0.10,
                               label="drilled sphere direct 120°")

    # ── Flanged cylinder with bolt holes ────────────────────────────

    def test_boolean_cut_flanged_bolt_pattern(self):
        """Flange with 6 bolt holes — boolean cut must handle the hole pattern."""
        import math as _m
        flange = (cq.Workplane("XY")
                  .circle(30).extrude(5)
                  .faces(">Z").workplane()
                  .cylinder(40, 12))
        # Drill 6 bolt holes in the flange ring
        for i in range(6):
            ang = _m.radians(60 * i)
            flange = (flange.faces("<Z").workplane()
                      .transformed(offset=(_m.cos(ang) * 22,
                                           _m.sin(ang) * 22, 0))
                      .hole(6))
        shape = flange.val().wrapped
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 90, tol=0.10,
                               label="flanged bolt pattern boolean 90°")

    def test_direct_cut_flanged_bolt_pattern(self):
        """Flange with 6 bolt holes — direct cut."""
        import math as _m
        flange = (cq.Workplane("XY")
                  .circle(30).extrude(5)
                  .faces(">Z").workplane()
                  .cylinder(40, 12))
        for i in range(6):
            ang = _m.radians(60 * i)
            flange = (flange.faces("<Z").workplane()
                      .transformed(offset=(_m.cos(ang) * 22,
                                           _m.sin(ang) * 22, 0))
                      .hole(6))
        shape = flange.val().wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 90, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 90, tol=0.10,
                               label="flanged bolt pattern direct 90°")

    # ── Nested concentric tubes (3 layers) ──────────────────────────

    def test_boolean_cut_nested_tubes(self):
        """Three concentric tubes assembled and cut as a compound."""
        outer = cq.Workplane("XY").cylinder(50, 25).faces(">Z").workplane().hole(44)
        mid   = cq.Workplane("XY").cylinder(50, 18).faces(">Z").workplane().hole(30)
        inner = cq.Workplane("XY").cylinder(50, 10).faces(">Z").workplane().hole(14)
        parts = [(outer, "outer_1"), (mid, "mid_1"), (inner, "inner_1")]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for name, shape, loc, rgb, *_rest in info:
            builder.Add(compound, apply_location(shape, loc))
        cutter = make_cutter(self._get_bbox_vals(compound), 90, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)
        assert result is not None and not result.IsNull()
        # Volume must decrease
        assert self._get_volume(result) < self._get_volume(compound) * 0.85

    def test_direct_cut_nested_tubes(self):
        """Three concentric tubes — direct cut each individually."""
        shapes = []
        for r_out, r_in in [(25, 22), (18, 15), (10, 7)]:
            s = (cq.Workplane("XY").cylinder(50, r_out)
                 .faces(">Z").workplane().hole(r_in * 2)).val().wrapped
            shapes.append(s)
        axis_vec = AXIS_MAP["z"]
        for i, shape in enumerate(shapes):
            origin = self._origin_for(shape, axis_vec)
            result = cut_part_direct(shape, 90, axis_vec, origin)
            self._assert_valid_cut(shape, result, 90, tol=0.10,
                                   label=f"nested tube {i} direct 90°")

    # ── Torus (donut) — curved surface cuts ─────────────────────────

    def _make_torus(self):
        """Build a solid torus (major=20, minor=8) via OCP directly."""
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeTorus
        return BRepPrimAPI_MakeTorus(20, 8).Shape()

    def test_boolean_cut_torus(self):
        """Boolean cut through a torus (tangent intersection with cutter)."""
        shape = self._make_torus()
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 90, tol=0.10,
                               label="torus boolean 90°")

    def test_direct_cut_torus_270(self):
        """Direct 270° cut on a torus — large removal, curved geometry."""
        shape = self._make_torus()
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 270, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 270, tol=0.10,
                               label="torus direct 270°")

    # ── Stepped cylinder (multi-diameter) ───────────────────────────

    def test_boolean_cut_stepped_cylinder(self):
        """Stepped cylinder with 4 diameter changes."""
        base = cq.Workplane("XY").cylinder(10, 25)
        mid  = base.faces(">Z").workplane().cylinder(15, 18)
        neck = mid.faces(">Z").workplane().cylinder(8, 12)
        top  = neck.faces(">Z").workplane().cylinder(12, 20)
        shape = top.val().wrapped
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 90, tol=0.10,
                               label="stepped cylinder boolean 90°")

    def test_direct_cut_stepped_cylinder_180(self):
        """Direct 180° cut on a stepped cylinder."""
        base = cq.Workplane("XY").cylinder(10, 25)
        mid  = base.faces(">Z").workplane().cylinder(15, 18)
        neck = mid.faces(">Z").workplane().cylinder(8, 12)
        top  = neck.faces(">Z").workplane().cylinder(12, 20)
        shape = top.val().wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 180, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 180, tol=0.10,
                               label="stepped cylinder direct 180°")

    # ── Box with internal cavity (mold-like) ────────────────────────

    def _make_box_with_cavity(self):
        """Box with a spherical internal cavity (mold shape)."""
        box = cq.Workplane("XY").box(50, 50, 50)
        cavity = cq.Workplane("XY").sphere(15)
        return box.cut(cavity).val().wrapped

    def test_boolean_cut_box_with_cavity(self):
        """Box with a spherical internal cavity — boolean cut."""
        shape = self._make_box_with_cavity()
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 90, tol=0.10,
                               label="box with cavity boolean 90°")

    def test_direct_cut_box_with_cavity(self):
        """Direct cut on box with spherical internal cavity."""
        shape = self._make_box_with_cavity()
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 90, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 90, tol=0.10,
                               label="box with cavity direct 90°")

    # ── Hex nut (polygon with central bore) ─────────────────────────

    def test_boolean_cut_hex_nut(self):
        """Hexagonal nut — polygon extrusion with central bore."""
        nut = (cq.Workplane("XY")
               .polygon(6, 30).extrude(12)
               .faces(">Z").workplane().hole(14))
        shape = nut.val().wrapped
        for angle in [45, 90, 180]:
            cutter = make_cutter(self._get_bbox_vals(shape), angle, AXIS_MAP["z"])
            result = cut_assembly(shape, cutter)
            self._assert_valid_cut(shape, result, angle, tol=0.10,
                                   label=f"hex nut boolean {angle}°")

    def test_direct_cut_hex_nut(self):
        """Direct cut on a hex nut at multiple angles."""
        nut = (cq.Workplane("XY")
               .polygon(6, 30).extrude(12)
               .faces(">Z").workplane().hole(14))
        shape = nut.val().wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])
        for angle in [45, 90, 180]:
            result = cut_part_direct(shape, angle, AXIS_MAP["z"], origin)
            self._assert_valid_cut(shape, result, angle, tol=0.10,
                                   label=f"hex nut direct {angle}°")

    # ── Cone frustum ────────────────────────────────────────────────

    def test_boolean_cut_cone_frustum(self):
        """Truncated cone (frustum) — tapered surface boolean cut."""
        frustum = cq.Solid.makeCone(30, 10, 50,
                                    pnt=cq.Vector(0, 0, -25),
                                    dir=cq.Vector(0, 0, 1))
        shape = frustum.wrapped
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 90, tol=0.10,
                               label="cone frustum boolean 90°")

    def test_direct_cut_cone_frustum(self):
        """Direct cut on a truncated cone."""
        frustum = cq.Solid.makeCone(30, 10, 50,
                                    pnt=cq.Vector(0, 0, -25),
                                    dir=cq.Vector(0, 0, 1))
        shape = frustum.wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 120, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 120, tol=0.10,
                               label="cone frustum direct 120°")

    # ── Cylinder with helical groove (thread-like) ──────────────────

    def test_boolean_cut_grooved_cylinder(self):
        """Cylinder with 4 longitudinal grooves (keyway-like features)."""
        import math as _m
        cyl = cq.Workplane("XY").cylinder(60, 20)
        # Cut 4 longitudinal slots
        for i in range(4):
            ang = _m.radians(90 * i)
            cx = _m.cos(ang) * 18
            cy = _m.sin(ang) * 18
            slot = (cq.Workplane("XY")
                    .transformed(offset=(cx, cy, 0))
                    .box(4, 4, 60))
            cyl = cyl.cut(slot)
        shape = cyl.val().wrapped
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 90, tol=0.12,
                               label="grooved cylinder boolean 90°")

    def test_direct_cut_grooved_cylinder(self):
        """Direct cut on a grooved cylinder."""
        import math as _m
        cyl = cq.Workplane("XY").cylinder(60, 20)
        for i in range(4):
            ang = _m.radians(90 * i)
            cx = _m.cos(ang) * 18
            cy = _m.sin(ang) * 18
            slot = (cq.Workplane("XY")
                    .transformed(offset=(cx, cy, 0))
                    .box(4, 4, 60))
            cyl = cyl.cut(slot)
        shape = cyl.val().wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 90, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 90, tol=0.12,
                               label="grooved cylinder direct 90°")

    # ── L-shaped bracket ────────────────────────────────────────────

    def _make_l_bracket(self):
        """Build an L-shaped bracket by unioning two boxes."""
        arm1 = cq.Workplane("XY").box(40, 10, 50)
        arm2 = cq.Workplane("XY").transformed(offset=(-15, 15, 0)).box(10, 20, 50)
        return arm1.union(arm2).val().wrapped

    def test_boolean_cut_l_bracket(self):
        """L-shaped bracket — non-symmetric cross section."""
        shape = self._make_l_bracket()
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        assert result is not None and not result.IsNull()
        verts, faces = tessellate_shape(result, tolerance=0.1)
        assert len(verts) > 0 and len(faces) > 0

    def test_direct_cut_l_bracket(self):
        """Direct cut on an L-shaped bracket."""
        shape = self._make_l_bracket()
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 90, AXIS_MAP["z"], origin)
        assert result is not None and not result.IsNull()
        verts, faces = tessellate_shape(result, tolerance=0.1)
        assert len(verts) > 0 and len(faces) > 0

    # ── Full assembly: concentric + cut + export round-trip ─────────

    def test_full_pipeline_complex_assembly(self, tmp_dir):
        """End-to-end: build complex concentric assembly, cut, export, reimport."""
        # Outer: box with holes
        outer = (cq.Workplane("XY").box(80, 80, 30)
                 .faces(">Z").workplane()
                 .pushPoints([(20, 20), (-20, 20), (20, -20), (-20, -20)])
                 .hole(8))
        # Mid: hollow cylinder
        mid = (cq.Workplane("XY").cylinder(30, 25)
               .faces(">Z").workplane().hole(40))
        # Inner: hex nut
        inner = (cq.Workplane("XY").polygon(6, 16).extrude(30)
                 .faces(">Z").workplane().hole(8))

        parts = [(outer, "outer_1"), (mid, "mid_1"), (inner, "inner_1")]
        assy, info = stack_parts(parts, AXIS_MAP["z"], gap=0)

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for name, shape, loc, rgb, *_rest in info:
            builder.Add(compound, apply_location(shape, loc))

        # Boolean cut
        cutter = make_cutter(self._get_bbox_vals(compound), 90, AXIS_MAP["z"])
        result = cut_assembly(compound, cutter)
        assert result is not None and not result.IsNull()

        # Export and reimport
        out_path = os.path.join(tmp_dir, "complex_assembly_cut.step")
        export_shape_step(result, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 500

        # Reimport should succeed
        reimported = cq.importers.importStep(out_path)
        assert reimported is not None

    # ── Stress: many-hole plate ─────────────────────────────────────

    def test_boolean_cut_perforated_plate(self):
        """Plate with a 3x3 grid of holes — many boolean features."""
        plate = cq.Workplane("XY").box(60, 60, 8)
        for x in [-18, 0, 18]:
            for y in [-18, 0, 18]:
                plate = (plate.faces(">Z").workplane()
                         .transformed(offset=(x, y, 0))
                         .hole(8))
        shape = plate.val().wrapped
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 90, tol=0.12,
                               label="perforated plate boolean 90°")

    def test_direct_cut_perforated_plate(self):
        """Direct cut on a perforated plate."""
        plate = cq.Workplane("XY").box(60, 60, 8)
        for x in [-18, 0, 18]:
            for y in [-18, 0, 18]:
                plate = (plate.faces(">Z").workplane()
                         .transformed(offset=(x, y, 0))
                         .hole(8))
        shape = plate.val().wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 90, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 90, tol=0.12,
                               label="perforated plate direct 90°")

    # ── Off-axis cutting ────────────────────────────────────────────

    def test_complex_shape_x_axis_cut(self):
        """Cut a drilled sphere along the X axis instead of Z."""
        sphere = cq.Workplane("XY").sphere(25)
        bore = cq.Workplane("XY").cylinder(60, 5)
        shape = sphere.cut(bore).val().wrapped
        for method in ["boolean", "direct"]:
            if method == "boolean":
                cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["x"])
                result = cut_assembly(shape, cutter)
            else:
                origin = self._origin_for(shape, AXIS_MAP["x"])
                result = cut_part_direct(shape, 90, AXIS_MAP["x"], origin)
            assert result is not None and not result.IsNull(), (
                f"X-axis {method} cut failed"
            )
            verts, faces = tessellate_shape(result, tolerance=0.1)
            assert len(verts) > 0, f"X-axis {method} cut has no vertices"

    def test_complex_shape_y_axis_cut(self):
        """Cut a stepped cylinder along the Y axis."""
        base = cq.Workplane("XY").cylinder(10, 25)
        top  = base.faces(">Z").workplane().cylinder(20, 15)
        shape = top.val().wrapped
        for method in ["boolean", "direct"]:
            if method == "boolean":
                cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["y"])
                result = cut_assembly(shape, cutter)
            else:
                origin = self._origin_for(shape, AXIS_MAP["y"])
                result = cut_part_direct(shape, 90, AXIS_MAP["y"], origin)
            assert result is not None and not result.IsNull(), (
                f"Y-axis {method} cut failed"
            )
            verts, faces = tessellate_shape(result, tolerance=0.1)
            assert len(verts) > 0, f"Y-axis {method} cut has no vertices"

    # ── Edge-case angles on complex geometry ────────────────────────

    def test_thin_wedge_on_torus(self):
        """Very thin 15° cut on a torus — near-degenerate wedge."""
        shape = self._make_torus()
        cutter = make_cutter(self._get_bbox_vals(shape), 15, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        self._assert_valid_cut(shape, result, 15, tol=0.10,
                               label="torus thin wedge 15°")

    def test_near_full_removal_on_drilled_sphere(self):
        """330° cut on drilled sphere — nearly everything removed."""
        sphere = cq.Workplane("XY").sphere(25)
        bore = cq.Workplane("XY").cylinder(60, 5)
        shape = sphere.cut(bore).val().wrapped
        origin = self._origin_for(shape, AXIS_MAP["z"])
        result = cut_part_direct(shape, 330, AXIS_MAP["z"], origin)
        self._assert_valid_cut(shape, result, 330, tol=0.15,
                               label="drilled sphere direct 330°")

    # ── STL mesh cutting (mesh → solid → cut) ───────────────────────

    def test_boolean_cut_stl_mesh_box(self, tmp_dir):
        """Load an STL mesh, convert to solid, and boolean-cut it."""
        # Create a moderately complex STL
        obj = (cq.Workplane("XY").box(40, 40, 20)
               .faces(">Z").workplane().hole(12))
        stl_path = os.path.join(tmp_dir, "complex_mesh.stl")
        obj.val().exportStl(stl_path)

        wp, name = load_part(stl_path)
        shape = wp.val().wrapped
        cutter = make_cutter(self._get_bbox_vals(shape), 90, AXIS_MAP["z"])
        result = cut_assembly(shape, cutter)
        assert result is not None and not result.IsNull()
        verts, faces = tessellate_shape(result, tolerance=0.2)
        assert len(verts) > 0 and len(faces) > 0

    # ── Segment cuts on complex shapes ──────────────────────────────

    def test_segment_cut_on_hollow_cylinder(self):
        """Direct segment (a/b) splitting on a hollow cylinder."""
        tube = (cq.Workplane("XY").cylinder(50, 20)
                .faces(">Z").workplane().hole(30))
        shape = tube.val().wrapped
        orig_vol = self._get_volume(shape)
        origin = self._origin_for(shape, AXIS_MAP["z"])

        seg_a = cut_part_direct_segment(shape, 90, AXIS_MAP["z"], origin, "a")
        seg_b = cut_part_direct_segment(shape, 90, AXIS_MAP["z"], origin, "b")
        assert seg_a is not None and seg_b is not None

        vol_a = self._get_volume(seg_a)
        vol_b = self._get_volume(seg_b)
        # Combined should approximate 75% of original (90° removed)
        combined = vol_a + vol_b
        expected = orig_vol * (270.0 / 360.0)
        assert abs(combined - expected) / expected < 0.10, (
            f"Segment volumes {vol_a:.1f}+{vol_b:.1f}={combined:.1f} "
            f"vs expected {expected:.1f}"
        )
