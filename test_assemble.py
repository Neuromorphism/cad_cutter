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
    shape_extent,
    translate_shape,
    apply_location,
    load_part,
    pick_color,
    parse_part_name,
    stack_parts,
    make_cutter,
    cut_assembly,
    export_assembly_step,
    export_shape_step,
    tessellate_shape,
    _mesh_shell_to_solid,
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
        tier, levels = parse_part_name("outer_1")
        assert tier == "outer"
        assert levels == [1]

    def test_mid_2(self):
        tier, levels = parse_part_name("mid_2")
        assert tier == "mid"
        assert levels == [2]

    def test_inner_3(self):
        tier, levels = parse_part_name("inner_3")
        assert tier == "inner"
        assert levels == [3]

    def test_case_insensitive(self):
        tier, levels = parse_part_name("OUTER_1")
        assert tier == "outer"
        assert levels == [1]

    def test_mixed_case(self):
        tier, levels = parse_part_name("Inner_5")
        assert tier == "inner"
        assert levels == [5]

    def test_hyphen_separator(self):
        tier, levels = parse_part_name("outer-1")
        assert tier == "outer"
        assert levels == [1]

    def test_no_separator(self):
        tier, levels = parse_part_name("outer1")
        assert tier == "outer"
        assert levels == [1]

    def test_with_material_prefix(self):
        tier, levels = parse_part_name("steel_outer_2")
        assert tier == "outer"
        assert levels == [2]

    def test_unrecognized_name(self):
        tier, levels = parse_part_name("plate")
        assert tier is None
        assert levels is None

    def test_unrecognized_with_number(self):
        tier, levels = parse_part_name("flange_3")
        assert tier is None
        assert levels is None

    def test_multi_digit_level(self):
        tier, levels = parse_part_name("outer_12")
        assert tier == "outer"
        assert levels == [12]

    def test_multi_level_span(self):
        tier, levels = parse_part_name("inner_2_3")
        assert tier == "inner"
        assert levels == [2, 3]

    def test_multi_level_three_sections(self):
        tier, levels = parse_part_name("mid_1_2_3")
        assert tier == "mid"
        assert levels == [1, 2, 3]

    def test_multi_level_with_material(self):
        tier, levels = parse_part_name("inner_2_3_steel")
        assert tier == "inner"
        assert levels == [2, 3]

    def test_multi_level_hyphen(self):
        tier, levels = parse_part_name("outer-1-2")
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
        for name, shape, loc, rgb in info:
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

        for name, shape, loc, rgb in level1:
            moved = apply_location(shape, loc)
            _, _, zn, _, _, _ = get_bounding_box(moved)
            assert abs(zn - 0) < 0.5, f"{name} z_bottom={zn}, expected 0"

        for name, shape, loc, rgb in level2:
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
                    for n, s, l, _ in info}

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
        tier, levels = parse_part_name("inner_2_3_steel")
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
        for name, shape, loc, rgb in info:
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

        for name, shape, loc, rgb in info:
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
        for name, shape, loc, rgb in info:
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

        for name, shape, loc, rgb in info:
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

        for name, shape, loc, rgb in info:
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
        for name, shape, loc, rgb in info:
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
        for name, shape, loc, rgb in info:
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

        reimported = cq.importers.importStep(out_path)
        shape = reimported.val().wrapped
        assert not shape.IsNull()

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
        for n, s, l, c in info:
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
        for n, s, l, c in info:
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
        for n, s, l, c in info:
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
        for name, shape, loc, rgb in info:
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
        for name, shape, loc, rgb in info:
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
