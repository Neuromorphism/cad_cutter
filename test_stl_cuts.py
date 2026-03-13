"""
Tests for STL cutting using beholder.stl.

Validates that:
  1. beholder.stl loads correctly and converts to a valid solid
  2. Boolean cutting (default) produces correct output STL
  3. Direct geometry cutting (--cut-direct) produces correct output STL
  4. Both methods agree on the cut result
  5. A plane intersection at the center produces a 2D cross-section
  6. Cutting in half produces a surface that matches the cross-section
  7. The _mesh_shell_to_solid fix picks the largest solid from compounds

Run with:
  python3 -m pytest test_stl_cuts.py -v
"""

import os
import sys
import math
import tempfile
import shutil

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cadquery as cq
from assemble import (
    load_part,
    is_mesh_file,
    get_bounding_box,
    make_cutter,
    cut_assembly,
    cut_part_direct,
    cut_part_direct_mesh,
    cut_part_direct_mesh_segment,
    _cutter_params,
    tessellate_shape,
    _mesh_shell_to_solid,
    midscale_parts,
    AXIS_MAP,
)

from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.TopoDS import TopoDS_Compound
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section, BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCP.StlAPI import StlAPI_Writer, StlAPI_Reader
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopoDS import TopoDS_Shape


BEHOLDER_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beholder.stl")


# ============================================================================
# Helpers
# ============================================================================

def get_volume(shape):
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return props.Mass()


def get_bbox_vals(shape):
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox)
    return bbox.Get()


def origin_for(shape, axis_vec):
    bbox_vals = get_bbox_vals(shape)
    _, _, origin_pt, _ = _cutter_params(bbox_vals, axis_vec)
    return origin_pt


def export_stl(shape, filepath, tolerance=0.1):
    BRepMesh_IncrementalMesh(shape, tolerance, False, 0.5, True)
    writer = StlAPI_Writer()
    writer.Write(shape, filepath)


def load_stl_trimesh(filepath):
    import trimesh
    return trimesh.load(filepath, force="mesh")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="test_stl_cuts_")
    yield d
    shutil.rmtree(d)


@pytest.fixture
def beholder_shape():
    """Load beholder.stl and return the OCP solid shape."""
    wp, name = load_part(BEHOLDER_STL)
    return wp.val().wrapped


class TestMidscale:
    def test_midscale_xy_and_z_contacts(self):
        outer_shell = (
            cq.Workplane("XY")
            .box(10, 10, 2)
            .cut(cq.Workplane("XY").box(8, 8, 2))
            .translate((0, 0, 1))
            .val()
            .wrapped
        )
        outer_below = cq.Workplane("XY").box(10, 10, 2).translate((0, 0, -2)).val().wrapped
        outer_above = cq.Workplane("XY").box(10, 10, 2).translate((0, 0, 4)).val().wrapped
        mid_shape = cq.Workplane("XY").box(6, 6, 2).translate((0, 0, 1)).val().wrapped

        part_info = [
            ("outer_1", outer_shell, cq.Location(), (0.5, 0.5, 0.5), None, False),
            ("outer_2", outer_above, cq.Location(), (0.5, 0.5, 0.5), None, False),
            ("outer_0", outer_below, cq.Location(), (0.5, 0.5, 0.5), None, False),
            ("mid_1", mid_shape, cq.Location(), (0.2, 0.2, 0.8), None, False),
        ]

        scaled = midscale_parts(part_info, debug=False)
        scaled_mid = [e for e in scaled if e[0] == "mid_1"][0][1]

        b0 = get_bbox_vals(mid_shape)
        b1 = get_bbox_vals(scaled_mid)

        assert (b1[3] - b1[0]) > (b0[3] - b0[0])
        assert (b1[5] - b1[2]) > (b0[5] - b0[2])
        assert b1[2] <= -1.0 + 1e-4
        assert b1[5] >= 3.0 - 1e-4



# ============================================================================
# Test: beholder.stl loading
# ============================================================================

class TestBeholderLoading:
    """Verify that beholder.stl loads correctly as a valid solid."""

    def test_loads_successfully(self):
        wp, name = load_part(BEHOLDER_STL)
        assert name == "beholder"
        shape = wp.val().wrapped
        assert shape is not None
        assert not shape.IsNull()

    def test_is_solid(self, beholder_shape):
        assert beholder_shape.ShapeType().name == "TopAbs_SOLID"

    def test_has_correct_extents(self, beholder_shape):
        bb = get_bbox_vals(beholder_shape)
        x_ext = bb[3] - bb[0]
        y_ext = bb[4] - bb[1]
        z_ext = bb[5] - bb[2]
        assert x_ext > 50, f"X extent too small: {x_ext}"
        assert y_ext > 70, f"Y extent too small: {y_ext}"
        assert z_ext > 70, f"Z extent too small: {z_ext}"

    def test_has_positive_volume(self, beholder_shape):
        vol = get_volume(beholder_shape)
        assert vol > 1000, f"Volume too small: {vol}"

    def test_has_faces(self, beholder_shape):
        exp = TopExp_Explorer(beholder_shape, TopAbs_FACE)
        count = 0
        while exp.More():
            count += 1
            exp.Next()
        assert count > 100, f"Expected many faces, got {count}"


# ============================================================================
# Test: _mesh_shell_to_solid picks largest solid
# ============================================================================

class TestMeshShellToSolidFix:
    """Verify that _mesh_shell_to_solid picks the largest solid when
    sewing produces multiple shells (as happens with non-watertight meshes)."""

    def test_beholder_gets_full_size_solid(self):
        """The beholder.stl solid should have the full bounding box,
        not a tiny fragment from the first shell."""
        reader = StlAPI_Reader()
        raw = TopoDS_Shape()
        reader.Read(raw, BEHOLDER_STL)

        solid = _mesh_shell_to_solid(raw)
        bb = get_bbox_vals(solid)
        x_ext = bb[3] - bb[0]
        y_ext = bb[4] - bb[1]
        z_ext = bb[5] - bb[2]

        assert x_ext > 50, f"Solid X extent {x_ext:.2f} too small (should be ~63)"
        assert y_ext > 70, f"Solid Y extent {y_ext:.2f} too small (should be ~79)"
        assert z_ext > 70, f"Solid Z extent {z_ext:.2f} too small (should be ~79)"

    def test_volume_matches_trimesh(self):
        """The OCP solid volume should approximately match trimesh's volume."""
        import trimesh
        tm = trimesh.load(BEHOLDER_STL, force="mesh")
        if not tm.is_watertight:
            trimesh.repair.fill_holes(tm)
            trimesh.repair.fix_normals(tm)

        wp, _ = load_part(BEHOLDER_STL)
        ocp_vol = get_volume(wp.val().wrapped)

        # Trimesh volume may differ due to repair, but should be same order
        assert abs(ocp_vol - abs(tm.volume)) / max(ocp_vol, abs(tm.volume)) < 0.15, (
            f"OCP volume {ocp_vol:.1f} vs trimesh {abs(tm.volume):.1f}"
        )


# ============================================================================
# Test: Boolean cut (default method)
# ============================================================================

class TestBooleanCut:
    """Cut beholder.stl with the boolean (default) method."""

    def test_cut_90_produces_result(self, beholder_shape, tmp_dir):
        bbox_vals = get_bbox_vals(beholder_shape)
        cutter = make_cutter(bbox_vals, 90, AXIS_MAP["z"])
        result = cut_assembly(beholder_shape, cutter)
        assert result is not None
        assert not result.IsNull()

    def test_cut_90_reduces_volume(self, beholder_shape):
        orig_vol = get_volume(beholder_shape)
        bbox_vals = get_bbox_vals(beholder_shape)
        cutter = make_cutter(bbox_vals, 90, AXIS_MAP["z"])
        result = cut_assembly(beholder_shape, cutter)

        cut_vol = get_volume(result)
        assert cut_vol < orig_vol, "Volume should decrease after cutting"
        assert cut_vol > 0, "Some volume should remain"
        ratio = cut_vol / orig_vol
        assert 0.5 < ratio < 0.95, f"90° cut should leave 50-95%, got {ratio*100:.1f}%"

    def test_cut_180_volume(self, beholder_shape):
        orig_vol = get_volume(beholder_shape)
        bbox_vals = get_bbox_vals(beholder_shape)
        cutter = make_cutter(bbox_vals, 180, AXIS_MAP["z"])
        result = cut_assembly(beholder_shape, cutter)

        cut_vol = get_volume(result)
        ratio = cut_vol / orig_vol
        assert 0.2 < ratio < 0.8, f"180° cut should leave 20-80%, got {ratio*100:.1f}%"

    def test_cut_exports_valid_stl(self, beholder_shape, tmp_dir):
        bbox_vals = get_bbox_vals(beholder_shape)
        cutter = make_cutter(bbox_vals, 90, AXIS_MAP["z"])
        result = cut_assembly(beholder_shape, cutter)

        stl_path = os.path.join(tmp_dir, "beholder_cut_90.stl")
        export_stl(result, stl_path)

        assert os.path.exists(stl_path)
        assert os.path.getsize(stl_path) > 0

        # Load back and verify it has geometry
        mesh = load_stl_trimesh(stl_path)
        assert len(mesh.vertices) > 100
        assert len(mesh.faces) > 100

    def test_cut_stl_within_original_bounds(self, beholder_shape, tmp_dir):
        """The cut STL bounding box should not exceed the original."""
        orig_bb = get_bbox_vals(beholder_shape)
        bbox_vals = orig_bb
        cutter = make_cutter(bbox_vals, 90, AXIS_MAP["z"])
        result = cut_assembly(beholder_shape, cutter)

        cut_bb = get_bbox_vals(result)
        tol = 1.0
        assert cut_bb[0] >= orig_bb[0] - tol, "Cut XMin exceeds original"
        assert cut_bb[1] >= orig_bb[1] - tol, "Cut YMin exceeds original"
        assert cut_bb[2] >= orig_bb[2] - tol, "Cut ZMin exceeds original"
        assert cut_bb[3] <= orig_bb[3] + tol, "Cut XMax exceeds original"
        assert cut_bb[4] <= orig_bb[4] + tol, "Cut YMax exceeds original"
        assert cut_bb[5] <= orig_bb[5] + tol, "Cut ZMax exceeds original"


# ============================================================================
# Test: Direct geometry cut (--cut-direct)
# ============================================================================

class TestDirectCut:
    """Cut beholder.stl with the direct geometry method."""

    def test_direct_cut_90_produces_result(self, beholder_shape):
        origin = origin_for(beholder_shape, AXIS_MAP["z"])
        result = cut_part_direct(beholder_shape, 90, AXIS_MAP["z"], origin)
        assert result is not None
        assert not result.IsNull()

    def test_direct_cut_90_reduces_volume(self, beholder_shape):
        orig_vol = get_volume(beholder_shape)
        origin = origin_for(beholder_shape, AXIS_MAP["z"])
        result = cut_part_direct(beholder_shape, 90, AXIS_MAP["z"], origin)

        cut_vol = get_volume(result)
        assert cut_vol < orig_vol, "Volume should decrease after cutting"
        assert cut_vol > 0, "Some volume should remain"
        ratio = cut_vol / orig_vol
        assert 0.5 < ratio < 0.95, f"90° cut should leave 50-95%, got {ratio*100:.1f}%"

    def test_direct_cut_180_volume(self, beholder_shape):
        orig_vol = get_volume(beholder_shape)
        origin = origin_for(beholder_shape, AXIS_MAP["z"])
        result = cut_part_direct(beholder_shape, 180, AXIS_MAP["z"], origin)

        cut_vol = get_volume(result)
        ratio = cut_vol / orig_vol
        assert 0.2 < ratio < 0.8, f"180° cut should leave 20-80%, got {ratio*100:.1f}%"

    def test_direct_cut_exports_valid_stl(self, beholder_shape, tmp_dir):
        origin = origin_for(beholder_shape, AXIS_MAP["z"])
        result = cut_part_direct(beholder_shape, 90, AXIS_MAP["z"], origin)

        stl_path = os.path.join(tmp_dir, "beholder_direct_cut_90.stl")
        export_stl(result, stl_path)

        assert os.path.exists(stl_path)
        assert os.path.getsize(stl_path) > 0

        mesh = load_stl_trimesh(stl_path)
        assert len(mesh.vertices) > 100
        assert len(mesh.faces) > 100


# ============================================================================
# Test: Boolean and direct methods agree
# ============================================================================

class TestMethodsAgree:
    """Both cut methods should produce similar volumes."""

    def test_90_degree_volumes_agree(self, beholder_shape):
        orig_vol = get_volume(beholder_shape)
        bbox_vals = get_bbox_vals(beholder_shape)
        axis_vec = AXIS_MAP["z"]

        # Boolean
        cutter = make_cutter(bbox_vals, 90, axis_vec)
        bool_result = cut_assembly(beholder_shape, cutter)
        bool_vol = get_volume(bool_result)

        # Direct
        origin = origin_for(beholder_shape, axis_vec)
        direct_result = cut_part_direct(beholder_shape, 90, axis_vec, origin)
        direct_vol = get_volume(direct_result)

        # Should be within 10% of each other
        diff = abs(bool_vol - direct_vol) / max(bool_vol, direct_vol)
        assert diff < 0.10, (
            f"Boolean vol={bool_vol:.1f} vs direct vol={direct_vol:.1f} "
            f"differ by {diff*100:.1f}%"
        )

    def test_180_degree_volumes_agree(self, beholder_shape):
        orig_vol = get_volume(beholder_shape)
        bbox_vals = get_bbox_vals(beholder_shape)
        axis_vec = AXIS_MAP["z"]

        cutter = make_cutter(bbox_vals, 180, axis_vec)
        bool_result = cut_assembly(beholder_shape, cutter)
        bool_vol = get_volume(bool_result)

        origin = origin_for(beholder_shape, axis_vec)
        direct_result = cut_part_direct(beholder_shape, 180, axis_vec, origin)
        direct_vol = get_volume(direct_result)

        diff = abs(bool_vol - direct_vol) / max(bool_vol, direct_vol)
        assert diff < 0.10, (
            f"Boolean vol={bool_vol:.1f} vs direct vol={direct_vol:.1f} "
            f"differ by {diff*100:.1f}%"
        )


# ============================================================================
# Test: Plane intersection gives 2D cross-section
# ============================================================================

class TestPlaneIntersection:
    """Intersect a horizontal plane with the center of beholder.stl
    to get a 2D cross-section."""

    def _center_z(self, shape):
        bb = get_bbox_vals(shape)
        return (bb[2] + bb[5]) / 2.0

    def test_section_produces_edges(self, beholder_shape):
        center_z = self._center_z(beholder_shape)
        plane = gp_Pln(gp_Pnt(0, 0, center_z), gp_Dir(0, 0, 1))
        section = BRepAlgoAPI_Section(beholder_shape, plane)
        section.Build()
        assert section.IsDone()

        section_shape = section.Shape()
        exp = TopExp_Explorer(section_shape, TopAbs_EDGE)
        count = 0
        while exp.More():
            count += 1
            exp.Next()
        assert count > 10, f"Expected many edges in cross-section, got {count}"

    def test_section_is_2d_at_center_z(self, beholder_shape):
        """The cross-section should lie entirely at z=center_z."""
        center_z = self._center_z(beholder_shape)
        plane = gp_Pln(gp_Pnt(0, 0, center_z), gp_Dir(0, 0, 1))
        section = BRepAlgoAPI_Section(beholder_shape, plane)
        section.Build()

        cs_bbox = Bnd_Box()
        BRepBndLib.Add_s(section.Shape(), cs_bbox)
        cs_bb = cs_bbox.Get()

        assert abs(cs_bb[2] - center_z) < 0.01, "Cross-section should be at center_z"
        assert abs(cs_bb[5] - center_z) < 0.01, "Cross-section should be at center_z"

    def test_section_spans_model_width(self, beholder_shape):
        """The cross-section XY extent should be similar to the model's XY extent."""
        bb = get_bbox_vals(beholder_shape)
        center_z = (bb[2] + bb[5]) / 2.0
        plane = gp_Pln(gp_Pnt(0, 0, center_z), gp_Dir(0, 0, 1))
        section = BRepAlgoAPI_Section(beholder_shape, plane)
        section.Build()

        cs_bbox = Bnd_Box()
        BRepBndLib.Add_s(section.Shape(), cs_bbox)
        cs_bb = cs_bbox.Get()

        model_x_ext = bb[3] - bb[0]
        model_y_ext = bb[4] - bb[1]
        cs_x_ext = cs_bb[3] - cs_bb[0]
        cs_y_ext = cs_bb[4] - cs_bb[1]

        # Cross-section should cover at least 50% of the model in XY
        assert cs_x_ext > model_x_ext * 0.5, (
            f"Cross-section X extent {cs_x_ext:.1f} too small vs model {model_x_ext:.1f}"
        )
        assert cs_y_ext > model_y_ext * 0.5, (
            f"Cross-section Y extent {cs_y_ext:.1f} too small vs model {model_y_ext:.1f}"
        )


# ============================================================================
# Test: Cut in half and verify cut surface matches cross-section
# ============================================================================

class TestCutSurfaceMatchesCrossSection:
    """Cut beholder.stl in half at center_z and verify the new planar
    surface matches the 2D cross-section from plane intersection."""

    def _get_center_z(self, shape):
        bb = get_bbox_vals(shape)
        return (bb[2] + bb[5]) / 2.0

    def _get_cross_section_bbox(self, shape, center_z):
        """Get the XY bounding box of the plane intersection at center_z."""
        plane = gp_Pln(gp_Pnt(0, 0, center_z), gp_Dir(0, 0, 1))
        section = BRepAlgoAPI_Section(shape, plane)
        section.Build()
        cs_bbox = Bnd_Box()
        BRepBndLib.Add_s(section.Shape(), cs_bbox)
        return cs_bbox.Get()

    def _cut_bottom_half(self, shape, center_z):
        """Cut away everything above center_z, returning the bottom half."""
        bb = get_bbox_vals(shape)
        margin = max(bb[3] - bb[0], bb[4] - bb[1]) * 2
        cutter_box = BRepPrimAPI_MakeBox(
            gp_Pnt(bb[0] - margin, bb[1] - margin, center_z),
            gp_Pnt(bb[3] + margin, bb[4] + margin, bb[5] + margin),
        ).Shape()
        cut_op = BRepAlgoAPI_Cut(shape, cutter_box)
        cut_op.Build()
        return cut_op.Shape()

    def test_cut_surface_x_range_matches(self, beholder_shape):
        """The X range of vertices on the cut plane should match the
        cross-section X range."""
        center_z = self._get_center_z(beholder_shape)
        cs_bb = self._get_cross_section_bbox(beholder_shape, center_z)

        bottom = self._cut_bottom_half(beholder_shape, center_z)
        verts, _ = tessellate_shape(bottom, tolerance=0.1)

        # Vertices near the cut plane
        cut_mask = np.abs(verts[:, 2] - center_z) < 0.5
        cut_verts = verts[cut_mask]
        assert len(cut_verts) > 10, "Should have vertices on the cut plane"

        tol = 2.0
        assert abs(cut_verts[:, 0].min() - cs_bb[0]) < tol, (
            f"Cut surface XMin {cut_verts[:, 0].min():.2f} vs "
            f"cross-section {cs_bb[0]:.2f}"
        )
        assert abs(cut_verts[:, 0].max() - cs_bb[3]) < tol, (
            f"Cut surface XMax {cut_verts[:, 0].max():.2f} vs "
            f"cross-section {cs_bb[3]:.2f}"
        )

    def test_cut_surface_y_range_matches(self, beholder_shape):
        """The Y range of vertices on the cut plane should match the
        cross-section Y range."""
        center_z = self._get_center_z(beholder_shape)
        cs_bb = self._get_cross_section_bbox(beholder_shape, center_z)

        bottom = self._cut_bottom_half(beholder_shape, center_z)
        verts, _ = tessellate_shape(bottom, tolerance=0.1)

        cut_mask = np.abs(verts[:, 2] - center_z) < 0.5
        cut_verts = verts[cut_mask]
        assert len(cut_verts) > 10

        tol = 2.0
        assert abs(cut_verts[:, 1].min() - cs_bb[1]) < tol, (
            f"Cut surface YMin {cut_verts[:, 1].min():.2f} vs "
            f"cross-section {cs_bb[1]:.2f}"
        )
        assert abs(cut_verts[:, 1].max() - cs_bb[4]) < tol, (
            f"Cut surface YMax {cut_verts[:, 1].max():.2f} vs "
            f"cross-section {cs_bb[4]:.2f}"
        )

    def test_bottom_half_volume_is_reasonable(self, beholder_shape):
        """The bottom half should contain a substantial portion of the volume.
        Note: beholder is asymmetric so the bottom half may contain most
        of the volume (the center_z is the bbox midpoint, not the volume
        centroid)."""
        center_z = self._get_center_z(beholder_shape)
        orig_vol = get_volume(beholder_shape)
        bottom = self._cut_bottom_half(beholder_shape, center_z)
        bottom_vol = get_volume(bottom)

        ratio = bottom_vol / orig_vol
        assert 0.01 < ratio < 1.0, (
            f"Bottom half should be >1% and <100% of original, got {ratio*100:.1f}%"
        )
        assert bottom_vol < orig_vol, "Bottom half should be smaller than original"

    def test_cut_surface_exported_stl_matches(self, beholder_shape, tmp_dir):
        """Export the bottom half as STL and verify the cut surface
        vertices match the cross-section bounds."""
        center_z = self._get_center_z(beholder_shape)
        cs_bb = self._get_cross_section_bbox(beholder_shape, center_z)

        bottom = self._cut_bottom_half(beholder_shape, center_z)
        stl_path = os.path.join(tmp_dir, "beholder_bottom.stl")
        export_stl(bottom, stl_path)

        mesh = load_stl_trimesh(stl_path)
        assert len(mesh.vertices) > 100

        # Vertices near the cut plane
        cut_verts = mesh.vertices[np.abs(mesh.vertices[:, 2] - center_z) < 0.5]
        assert len(cut_verts) > 10

        tol = 2.0
        assert abs(cut_verts[:, 0].min() - cs_bb[0]) < tol
        assert abs(cut_verts[:, 0].max() - cs_bb[3]) < tol
        assert abs(cut_verts[:, 1].min() - cs_bb[1]) < tol
        assert abs(cut_verts[:, 1].max() - cs_bb[4]) < tol


# ============================================================================
# Test: Multiple cut angles produce monotonically decreasing volume
# ============================================================================

class TestMonotonicVolume:
    """Larger cut angles should remove more material."""

    def test_volume_decreases_with_angle(self, beholder_shape):
        """Cutting at 90°, 180°, 270° should produce decreasing volumes."""
        orig_vol = get_volume(beholder_shape)
        bbox_vals = get_bbox_vals(beholder_shape)
        axis_vec = AXIS_MAP["z"]

        volumes = []
        for angle in [90, 180, 270]:
            cutter = make_cutter(bbox_vals, angle, axis_vec)
            result = cut_assembly(beholder_shape, cutter)
            if result is not None:
                volumes.append(get_volume(result))
            else:
                volumes.append(0)

        assert volumes[0] > volumes[1] > volumes[2], (
            f"Volumes should decrease: 90°={volumes[0]:.0f}, "
            f"180°={volumes[1]:.0f}, 270°={volumes[2]:.0f}"
        )


# ============================================================================
# Test: Native mesh direct-cut (trimesh boolean)
# ============================================================================

class TestNativeMeshDirectCut:
    """Tests for cut_part_direct_mesh() — native trimesh boolean cutting."""

    def test_mesh_cut_90_produces_result(self, beholder_shape):
        """90-degree native mesh cut should produce a non-null result."""
        origin = origin_for(beholder_shape, AXIS_MAP["z"])
        result = cut_part_direct_mesh(beholder_shape, 90, AXIS_MAP["z"], origin)
        assert result is not None
        assert not result.IsNull()

    def test_mesh_cut_90_reduces_volume(self, beholder_shape):
        """90-degree native mesh cut should reduce volume."""
        orig_vol = get_volume(beholder_shape)
        origin = origin_for(beholder_shape, AXIS_MAP["z"])
        result = cut_part_direct_mesh(beholder_shape, 90, AXIS_MAP["z"], origin)

        cut_vol = get_volume(result)
        assert cut_vol < orig_vol, "Volume should decrease after cutting"
        assert cut_vol > 0, "Some volume should remain"
        ratio = cut_vol / orig_vol
        assert 0.5 < ratio < 0.95, f"90° cut should leave 50-95%, got {ratio*100:.1f}%"

    def test_mesh_cut_180_volume(self, beholder_shape):
        """180-degree native mesh cut should remove roughly half."""
        orig_vol = get_volume(beholder_shape)
        origin = origin_for(beholder_shape, AXIS_MAP["z"])
        result = cut_part_direct_mesh(beholder_shape, 180, AXIS_MAP["z"], origin)

        cut_vol = get_volume(result)
        ratio = cut_vol / orig_vol
        assert 0.2 < ratio < 0.8, f"180° cut should leave 20-80%, got {ratio*100:.1f}%"

    def test_mesh_cut_exports_valid_stl(self, beholder_shape, tmp_dir):
        """Native mesh cut result should export to a valid STL."""
        origin = origin_for(beholder_shape, AXIS_MAP["z"])
        result = cut_part_direct_mesh(beholder_shape, 90, AXIS_MAP["z"], origin)

        stl_path = os.path.join(tmp_dir, "beholder_mesh_cut_90.stl")
        export_stl(result, stl_path)

        assert os.path.exists(stl_path)
        assert os.path.getsize(stl_path) > 0

        mesh = load_stl_trimesh(stl_path)
        assert len(mesh.vertices) > 100
        assert len(mesh.faces) > 100


# ============================================================================
# Test: Native mesh direct-cut agrees with parametric direct-cut
# ============================================================================

class TestNativeVsParametricDirectCut:
    """Both native cut methods should produce similar volumes on mesh data."""

    def test_90_degree_volumes_agree(self, beholder_shape):
        """Native mesh and parametric direct-cut volumes at 90° should agree."""
        orig_vol = get_volume(beholder_shape)
        axis_vec = AXIS_MAP["z"]
        origin = origin_for(beholder_shape, axis_vec)

        # Parametric (OCP Splitter)
        parametric_result = cut_part_direct(beholder_shape, 90, axis_vec, origin)
        parametric_vol = get_volume(parametric_result)

        # Native mesh (trimesh boolean)
        mesh_result = cut_part_direct_mesh(beholder_shape, 90, axis_vec, origin)
        mesh_vol = get_volume(mesh_result)

        # Should be within 15% of each other (mesh boolean is approximate)
        diff = abs(parametric_vol - mesh_vol) / max(parametric_vol, mesh_vol)
        assert diff < 0.15, (
            f"Parametric vol={parametric_vol:.1f} vs mesh vol={mesh_vol:.1f} "
            f"differ by {diff*100:.1f}%"
        )


# ============================================================================
# Test: Native parametric direct-cut on CAD geometry
# ============================================================================

class TestNativeParametricDirectCut:
    """Verify that parametric (OCP Splitter) direct-cut works on CAD geometry."""

    def _get_volume(self, shape):
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        return props.Mass()

    def _origin_for(self, shape, axis_vec):
        bbox_vals = get_bbox_vals(shape)
        _, _, origin_pt, _ = _cutter_params(bbox_vals, axis_vec)
        return origin_pt

    def test_parametric_cut_90_cylinder(self):
        """90° parametric cut of a CadQuery cylinder removes ~25%."""
        cyl = cq.Workplane("XY").cylinder(30, 8).val().wrapped
        orig_vol = self._get_volume(cyl)
        origin = self._origin_for(cyl, AXIS_MAP["z"])

        result = cut_part_direct(cyl, 90, AXIS_MAP["z"], origin)
        assert result is not None
        result_vol = self._get_volume(result)
        assert abs(result_vol / orig_vol - 0.75) < 0.02

    def test_parametric_cut_90_box(self):
        """90° parametric cut of a CadQuery box removes ~25%."""
        box = cq.Workplane("XY").box(20, 20, 10).val().wrapped
        orig_vol = self._get_volume(box)
        origin = self._origin_for(box, AXIS_MAP["z"])

        result = cut_part_direct(box, 90, AXIS_MAP["z"], origin)
        assert result is not None
        result_vol = self._get_volume(result)
        assert abs(result_vol / orig_vol - 0.75) < 0.02


# ============================================================================
# Test: is_mesh_file correctly detects mesh vs CAD files
# ============================================================================

class TestIsMeshFile:
    """Verify that is_mesh_file correctly classifies file types."""

    def test_stl_is_mesh(self):
        assert is_mesh_file("model.stl") is True
        assert is_mesh_file("model.STL") is True

    def test_obj_is_mesh(self):
        assert is_mesh_file("model.obj") is True

    def test_ply_is_mesh(self):
        assert is_mesh_file("model.ply") is True

    def test_3mf_is_mesh(self):
        assert is_mesh_file("model.3mf") is True

    def test_step_is_not_mesh(self):
        assert is_mesh_file("model.step") is False
        assert is_mesh_file("model.STEP") is False

    def test_iges_is_not_mesh(self):
        assert is_mesh_file("model.iges") is False

    def test_brep_is_not_mesh(self):
        assert is_mesh_file("model.brep") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
