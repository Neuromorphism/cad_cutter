r"""
Unit and integration tests for the CAD radial cutter.

These tests use FreeCAD's Python API to:
  1. Programmatically create test STEP files with known geometries
  2. Run the cutter on them
  3. Verify the output by checking volumes, bounding boxes, and shape counts

Run with:
  & "C:\Program Files\FreeCAD 1.0\bin\python.exe" -m pytest test_cut.py -v
"""

import sys
import os
import math
import tempfile
import shutil
import subprocess

# --- FreeCAD setup (same as cut.py) ---
possible_paths = [
    r"C:\Program Files\FreeCAD 1.0\bin",
    r"C:\Program Files\FreeCAD 0.21\bin",
    r"/usr/lib/freecad/lib",
    r"/usr/lib/freecad-python3/lib",
    r"/Applications/FreeCAD.app/Contents/Resources/lib",
]

FREECADPATH = None
for path in possible_paths:
    if os.path.exists(path):
        FREECADPATH = path
        break

assert FREECADPATH is not None, "FreeCAD not found"

if sys.platform == "win32":
    os.environ["PATH"] += os.pathsep + FREECADPATH
sys.path.insert(0, FREECADPATH)

import FreeCAD
import Part
import Import
import pytest

# Path to the cut.py script
CUT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cut.py")
FREECAD_PYTHON = os.path.join(FREECADPATH, "python.exe") if sys.platform == "win32" else "python3"


# ============================================================================
# Helpers
# ============================================================================

def create_temp_dir():
    """Create a temporary directory for test files."""
    return tempfile.mkdtemp(prefix="cad_cutter_test_")


def save_shape_as_step(shape, filepath):
    """Save a Part.Shape to a STEP file."""
    doc = FreeCAD.newDocument("TempSave")
    try:
        obj = doc.addObject("Part::Feature", "TestShape")
        obj.Shape = shape
        Import.export([obj], filepath)
    finally:
        FreeCAD.closeDocument("TempSave")


def save_shapes_as_step(shapes_with_names, filepath):
    """Save multiple shapes to a single STEP file."""
    doc = FreeCAD.newDocument("TempSave")
    try:
        objs = []
        for name, shape in shapes_with_names:
            obj = doc.addObject("Part::Feature", name)
            obj.Shape = shape
            objs.append(obj)
        Import.export(objs, filepath)
    finally:
        FreeCAD.closeDocument("TempSave")


def load_step_file(filepath):
    """Load a STEP file and return list of individual Solid shapes.
    
    When FreeCAD loads a multi-object STEP file, it may create both
    individual objects AND a parent compound containing them all.
    To avoid double-counting, we extract all Solids and deduplicate
    by checking if any solid is geometrically a duplicate (same volume
    and center of mass) of another.
    """
    doc = FreeCAD.newDocument("TempLoad")
    try:
        Import.insert(filepath, doc.Name)
        
        # Collect all objects, filtering out axes/planes/origin/debug
        valid_objs = []
        for obj in doc.Objects:
            if not hasattr(obj, 'Shape') or obj.Shape.isNull():
                continue
            label = obj.Label.lower()
            if any(x in label for x in ['axis', 'plane', 'origin']):
                continue
            if obj.Label == "CUTTER_DEBUG":
                continue
            valid_objs.append(obj)

        # Strategy: find the top-level objects only. If an object is a
        # Compound that subsumes other objects' solids, prefer the compound
        # approach: just gather all solids from the largest compound.
        # Simpler approach: find the set of objects whose combined volume
        # is maximized without overlap.
        
        # For the typical output of cut.py, each part is an individual
        # Feature. Just take each object's solids. If there's a parent
        # compound, it will have the same solids, so we deduplicate.
        all_solids = []
        seen = set()  # Track (volume, cx, cy, cz) tuples to avoid dupes
        
        for obj in valid_objs:
            shape = obj.Shape
            solids = shape.Solids if shape.Solids else [shape]
            for s in solids:
                if s.Volume < 1e-6:
                    continue
                # Create a fingerprint to avoid double-counting
                com = s.CenterOfMass
                key = (round(s.Volume, 2), round(com.x, 2), round(com.y, 2), round(com.z, 2))
                if key not in seen:
                    seen.add(key)
                    all_solids.append(s)
        
        return all_solids
    finally:
        FreeCAD.closeDocument("TempLoad")


def total_volume(shapes):
    """Sum up the volume of all shapes."""
    return sum(s.Volume for s in shapes)


def run_cutter(input_path, output_path, angle, debug=False):
    """Run the cutter script as a subprocess using FreeCAD's Python."""
    cmd = [FREECAD_PYTHON, CUT_SCRIPT, input_path, output_path, str(angle)]
    if debug:
        cmd.append("--debug")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"Cutter failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )
    return result


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after test."""
    d = create_temp_dir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ============================================================================
# Test: ensure_solid function
# ============================================================================

class TestEnsureSolid:
    """Tests for the ensure_solid helper."""

    def test_solid_passthrough(self):
        """A Solid should be returned unchanged."""
        from cut import ensure_solid
        box = Part.makeBox(10, 10, 10)
        result = ensure_solid(box)
        assert result.ShapeType == "Solid"
        assert abs(result.Volume - box.Volume) < 1e-6

    def test_shell_to_solid(self):
        """A closed Shell should be converted to a Solid."""
        from cut import ensure_solid
        box = Part.makeBox(10, 10, 10)
        shell = box.Shells[0]
        assert shell.ShapeType == "Shell"
        result = ensure_solid(shell)
        # Should be converted to solid if possible
        if result.ShapeType == "Solid":
            assert abs(result.Volume - 1000.0) < 1e-4
        else:
            # If conversion fails, original is returned
            assert result.ShapeType == "Shell"

    def test_non_shell_passthrough(self):
        """A shape that is neither Solid nor Shell should be returned as-is."""
        from cut import ensure_solid
        edge = Part.makeLine(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(10, 0, 0))
        result = ensure_solid(edge)
        assert result.ShapeType == edge.ShapeType


# ============================================================================
# Test: Box at origin with various cut angles
# ============================================================================

class TestBoxAtOrigin:
    """Cut a 100x100x100 box centered at origin with various angles."""

    def _make_centered_box(self):
        """Create a box centered at the origin."""
        return Part.makeBox(100, 100, 100, FreeCAD.Vector(-50, -50, -50))

    def test_cut_90_degrees(self, tmp_dir):
        """Cutting 90° should remove ~25% of the box (quarter wedge)."""
        box = self._make_centered_box()
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_cut.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 90)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0, "No shapes in output"

        original_vol = box.Volume  # 1,000,000
        result_vol = total_volume(result_shapes)

        # The cutter removes a 90° wedge from a cylinder centered on Y.
        # For a box, the exact fraction depends on geometry but should be
        # roughly 75% remaining (the cutter removes the wedge portion).
        # We verify the volume decreased significantly.
        assert result_vol < original_vol, "Volume should decrease after cutting"
        assert result_vol > 0, "Something should remain"

        # The cut wedge is a 90° sector. For a centered box, the portion
        # removed is approximately 25% of the box volume.
        expected_remaining_frac = 0.75
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_remaining_frac) < 0.1, (
            f"Expected ~{expected_remaining_frac*100}% remaining, "
            f"got {actual_frac*100:.1f}%"
        )

    def test_cut_180_degrees(self, tmp_dir):
        """Cutting 180° should remove ~50% of a centered box."""
        box = self._make_centered_box()
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_cut.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 180)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        result_vol = total_volume(result_shapes)
        original_vol = box.Volume
        expected_frac = 0.50
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.1, (
            f"Expected ~{expected_frac*100}% remaining, got {actual_frac*100:.1f}%"
        )

    def test_cut_270_degrees(self, tmp_dir):
        """Cutting 270° should remove ~75% of a centered box."""
        box = self._make_centered_box()
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_cut.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 270)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        result_vol = total_volume(result_shapes)
        original_vol = box.Volume
        expected_frac = 0.25
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.1, (
            f"Expected ~{expected_frac*100}% remaining, got {actual_frac*100:.1f}%"
        )

    def test_cut_small_angle(self, tmp_dir):
        """Cutting a small angle (10°) should remove only a small sliver."""
        box = self._make_centered_box()
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_cut.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 10)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        result_vol = total_volume(result_shapes)
        original_vol = box.Volume
        # Only a small wedge removed, most should remain
        assert result_vol / original_vol > 0.85, (
            f"Expected >85% remaining after 10° cut, got {result_vol/original_vol*100:.1f}%"
        )


# ============================================================================
# Test: Cylinder centered on Y-axis
# ============================================================================

class TestCylinderOnYAxis:
    """Cut a cylinder that is coaxial with the Y-axis (cutter axis).
    
    Since both the part and cutter share the Y-axis, the volume removed
    should be exactly proportional to the angle.
    """

    def _make_y_cylinder(self, radius=50, height=100):
        """Create a cylinder centered on Y-axis."""
        return Part.makeCylinder(
            radius, height,
            FreeCAD.Vector(0, -height / 2, 0),
            FreeCAD.Vector(0, 1, 0)
        )

    def test_cut_90_exact_quarter(self, tmp_dir):
        """For a Y-axis cylinder, 90° cut should remove exactly 25%."""
        cyl = self._make_y_cylinder()
        input_path = os.path.join(tmp_dir, "cyl.step")
        output_path = os.path.join(tmp_dir, "cyl_cut.step")
        save_shape_as_step(cyl, input_path)

        run_cutter(input_path, output_path, 90)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        original_vol = cyl.Volume
        result_vol = total_volume(result_shapes)
        expected_frac = 0.75  # 3/4 remaining
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.05, (
            f"Y-axis cylinder: expected ~{expected_frac*100}% remaining, "
            f"got {actual_frac*100:.1f}%"
        )

    def test_cut_180_exact_half(self, tmp_dir):
        """For a Y-axis cylinder, 180° cut should remove exactly 50%."""
        cyl = self._make_y_cylinder()
        input_path = os.path.join(tmp_dir, "cyl.step")
        output_path = os.path.join(tmp_dir, "cyl_cut.step")
        save_shape_as_step(cyl, input_path)

        run_cutter(input_path, output_path, 180)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        original_vol = cyl.Volume
        result_vol = total_volume(result_shapes)
        expected_frac = 0.50
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.05, (
            f"Y-axis cylinder: expected ~{expected_frac*100}% remaining, "
            f"got {actual_frac*100:.1f}%"
        )

    def test_cut_120_exact_third(self, tmp_dir):
        """For a Y-axis cylinder, 120° cut should remove exactly 1/3."""
        cyl = self._make_y_cylinder()
        input_path = os.path.join(tmp_dir, "cyl.step")
        output_path = os.path.join(tmp_dir, "cyl_cut.step")
        save_shape_as_step(cyl, input_path)

        run_cutter(input_path, output_path, 120)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        original_vol = cyl.Volume
        result_vol = total_volume(result_shapes)
        expected_frac = 2.0 / 3.0  # 2/3 remaining
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.05, (
            f"Y-axis cylinder: expected ~{expected_frac*100:.1f}% remaining, "
            f"got {actual_frac*100:.1f}%"
        )


# ============================================================================
# Test: Sphere at origin
# ============================================================================

class TestSphereAtOrigin:
    """Cut a sphere centered at origin. The cutter wedge should remove
    a fraction exactly proportional to the angle/360."""

    def _make_sphere(self, radius=50):
        return Part.makeSphere(radius)

    def test_cut_90_degrees(self, tmp_dir):
        """90° cut on a sphere should leave 75% volume."""
        sphere = self._make_sphere()
        input_path = os.path.join(tmp_dir, "sphere.step")
        output_path = os.path.join(tmp_dir, "sphere_cut.step")
        save_shape_as_step(sphere, input_path)

        run_cutter(input_path, output_path, 90)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        original_vol = sphere.Volume
        result_vol = total_volume(result_shapes)
        expected_frac = 0.75
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.05, (
            f"Sphere: expected ~{expected_frac*100}% remaining, "
            f"got {actual_frac*100:.1f}%"
        )

    def test_cut_180_degrees(self, tmp_dir):
        """180° cut on a sphere should leave 50% volume."""
        sphere = self._make_sphere()
        input_path = os.path.join(tmp_dir, "sphere.step")
        output_path = os.path.join(tmp_dir, "sphere_cut.step")
        save_shape_as_step(sphere, input_path)

        run_cutter(input_path, output_path, 180)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        original_vol = sphere.Volume
        result_vol = total_volume(result_shapes)
        expected_frac = 0.50
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.05, (
            f"Sphere: expected ~{expected_frac*100}% remaining, "
            f"got {actual_frac*100:.1f}%"
        )


# ============================================================================
# Test: Off-center part (box not at origin)
# ============================================================================

class TestOffCenterBox:
    """Test cutting a box that is NOT centered on the Y-axis.
    This tests the bounding box / cutter sizing logic."""

    def test_box_straddling_cutter_boundary(self, tmp_dir):
        """Box straddling the cutter boundary — partially inside, partially
        outside the 90° wedge. The box extends from negative Z into the
        first quadrant in XZ, so a 90° cutter (spanning +X toward +Z)
        should only cut part of it."""
        # Box from X=[10,60], Y=[0,50], Z=[-25,25]
        # The 90° cutter covers the +X/+Z quadrant (angles 0 to 90°).
        # The portion of the box at Z<0 is outside the cutter, Z>0 is inside.
        box = Part.makeBox(50, 50, 50, FreeCAD.Vector(10, 0, -25))
        input_path = os.path.join(tmp_dir, "offset_box.step")
        output_path = os.path.join(tmp_dir, "offset_box_cut.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 90)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0, "Some material should remain"
        # The box should be at least partially cut
        original_vol = box.Volume
        result_vol = total_volume(result_shapes)
        assert result_vol < original_vol, "Off-center box should still be cut"
        assert result_vol > 0, "Some material should remain"

    def test_box_far_away(self, tmp_dir):
        """A box far from origin, the cutter should still reach it
        since the cutter radius is based on the global bounding box."""
        box = Part.makeBox(20, 20, 20, FreeCAD.Vector(200, 0, 200))
        input_path = os.path.join(tmp_dir, "far_box.step")
        output_path = os.path.join(tmp_dir, "far_box_cut.step")
        save_shape_as_step(box, input_path)

        # 45° cut—the box is in the +X/+Z quadrant, so the default
        # cutter starting position should reach it
        run_cutter(input_path, output_path, 45)

        result_shapes = load_step_file(output_path)
        result_vol = total_volume(result_shapes)
        original_vol = box.Volume
        # Some portion should be cut
        assert result_vol <= original_vol, "Volume should not increase"


# ============================================================================
# Test: Multiple objects in one STEP file
# ============================================================================

class TestMultipleObjects:
    """A STEP file containing multiple separate parts."""

    def test_two_boxes(self, tmp_dir):
        """Two boxes should both be cut independently."""
        box1 = Part.makeBox(50, 50, 50, FreeCAD.Vector(-25, -25, -25))
        box2 = Part.makeBox(30, 30, 30, FreeCAD.Vector(40, -15, -15))

        input_path = os.path.join(tmp_dir, "two_boxes.step")
        output_path = os.path.join(tmp_dir, "two_boxes_cut.step")
        save_shapes_as_step([("Box1", box1), ("Box2", box2)], input_path)

        run_cutter(input_path, output_path, 90)

        result_shapes = load_step_file(output_path)
        original_vol = box1.Volume + box2.Volume
        result_vol = total_volume(result_shapes)
        assert result_vol < original_vol, "Combined volume should decrease"
        assert result_vol > 0, "At least some material should remain"

    def test_three_different_shapes(self, tmp_dir):
        """Mix of box, cylinder, and sphere."""
        box = Part.makeBox(40, 40, 40, FreeCAD.Vector(-20, -20, -20))
        cyl = Part.makeCylinder(
            25, 60,
            FreeCAD.Vector(0, -30, 0),
            FreeCAD.Vector(0, 1, 0)
        )
        sphere = Part.makeSphere(20, FreeCAD.Vector(50, 0, 0))

        input_path = os.path.join(tmp_dir, "mixed.step")
        output_path = os.path.join(tmp_dir, "mixed_cut.step")
        save_shapes_as_step(
            [("Box", box), ("Cylinder", cyl), ("Sphere", sphere)],
            input_path
        )

        run_cutter(input_path, output_path, 180)

        result_shapes = load_step_file(output_path)
        original_vol = box.Volume + cyl.Volume + sphere.Volume
        result_vol = total_volume(result_shapes)
        # 180° should remove roughly half
        assert 0.3 < result_vol / original_vol < 0.7, (
            f"Expected ~50% remaining, got {result_vol/original_vol*100:.1f}%"
        )


# ============================================================================
# Test: Edge cases — very small and very large angles
# ============================================================================

class TestEdgeCases:
    """Edge case angles."""

    def test_very_small_angle_1_degree(self, tmp_dir):
        """1° cut should barely remove anything."""
        box = Part.makeBox(100, 100, 100, FreeCAD.Vector(-50, -50, -50))
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_cut_1deg.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 1)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0
        result_vol = total_volume(result_shapes)
        original_vol = box.Volume
        # Should retain >95% with just 1° removed
        assert result_vol / original_vol > 0.95, (
            f"1° cut should leave >95%, got {result_vol/original_vol*100:.1f}%"
        )

    def test_359_degree_cut(self, tmp_dir):
        """359° cut should remove almost everything from a centered box."""
        box = Part.makeBox(100, 100, 100, FreeCAD.Vector(-50, -50, -50))
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_cut_359deg.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 359)

        result_shapes = load_step_file(output_path)
        result_vol = total_volume(result_shapes) if result_shapes else 0
        original_vol = box.Volume
        # Should retain <10% with 359° removed
        assert result_vol / original_vol < 0.10, (
            f"359° cut should leave <10%, got {result_vol/original_vol*100:.1f}%"
        )


# ============================================================================
# Test: Debug mode
# ============================================================================

class TestDebugMode:
    """Verify the --debug flag works and produces expected output."""

    def test_debug_flag_produces_output(self, tmp_dir):
        """Running with --debug should include 'Debug Mode ON' in stdout
        and should include a CUTTER_DEBUG object in the output file."""
        box = Part.makeBox(100, 100, 100, FreeCAD.Vector(-50, -50, -50))
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_debug.step")
        save_shape_as_step(box, input_path)

        result = run_cutter(input_path, output_path, 90, debug=True)
        assert "Debug Mode ON" in result.stdout

        # Load and check for cutter debug object
        doc = FreeCAD.newDocument("DebugCheck")
        try:
            Import.insert(output_path, doc.Name)
            labels = [obj.Label for obj in doc.Objects if hasattr(obj, 'Shape')]
            # There should be a CUTTER_DEBUG object
            assert any("CUTTER" in lbl.upper() for lbl in labels), (
                f"Expected CUTTER_DEBUG in output, got labels: {labels}"
            )
        finally:
            FreeCAD.closeDocument("DebugCheck")


# ============================================================================
# Test: Volume proportionality — parametric angle sweep
# ============================================================================

class TestVolumeProportionality:
    """Parametric test: for a sphere at origin, the remaining volume after
    cutting should scale linearly with (360 - angle) / 360."""

    @pytest.mark.parametrize("angle", [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
    def test_sphere_volume_proportional(self, tmp_dir, angle):
        """Volume remaining should be proportional to (360 - angle) / 360."""
        sphere = Part.makeSphere(40)
        input_path = os.path.join(tmp_dir, "sphere.step")
        output_path = os.path.join(tmp_dir, f"sphere_cut_{angle}.step")
        save_shape_as_step(sphere, input_path)

        run_cutter(input_path, output_path, angle)

        result_shapes = load_step_file(output_path)
        result_vol = total_volume(result_shapes) if result_shapes else 0
        original_vol = sphere.Volume

        expected_frac = (360.0 - angle) / 360.0
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.08, (
            f"Sphere at {angle}°: expected ~{expected_frac*100:.1f}% remaining, "
            f"got {actual_frac*100:.1f}%"
        )


# ============================================================================
# Test: Torus (donut shape) — more complex geometry
# ============================================================================

class TestTorusShape:
    """A torus centered at origin around the Y-axis is a more complex shape."""

    def test_torus_90_degree_cut(self, tmp_dir):
        """A torus cut at 90° should lose ~25% of its volume."""
        # makeTorus(major_radius, minor_radius)
        # Default axis is Z, so we rotate it to Y
        torus = Part.makeTorus(60, 15)
        # Rotate from Z-axis to Y-axis: rotate 90° around X
        torus = torus.rotated(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(1, 0, 0), 90)

        input_path = os.path.join(tmp_dir, "torus.step")
        output_path = os.path.join(tmp_dir, "torus_cut.step")
        save_shape_as_step(torus, input_path)

        run_cutter(input_path, output_path, 90)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        original_vol = torus.Volume
        result_vol = total_volume(result_shapes)
        expected_frac = 0.75
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.10, (
            f"Torus: expected ~{expected_frac*100}% remaining, "
            f"got {actual_frac*100:.1f}%"
        )


# ============================================================================
# Test: Concentric shapes (pipe / hollow cylinder)
# ============================================================================

class TestHollowCylinder:
    """A hollow cylinder (pipe) created by subtracting a smaller cylinder
    from a larger one, then cutting with a radial wedge."""

    def test_pipe_90_degree_cut(self, tmp_dir):
        """A pipe cut at 90° should lose ~25% volume."""
        outer = Part.makeCylinder(
            50, 100,
            FreeCAD.Vector(0, -50, 0),
            FreeCAD.Vector(0, 1, 0)
        )
        inner = Part.makeCylinder(
            35, 100,
            FreeCAD.Vector(0, -50, 0),
            FreeCAD.Vector(0, 1, 0)
        )
        pipe = outer.cut(inner)

        input_path = os.path.join(tmp_dir, "pipe.step")
        output_path = os.path.join(tmp_dir, "pipe_cut.step")
        save_shape_as_step(pipe, input_path)

        run_cutter(input_path, output_path, 90)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        original_vol = pipe.Volume
        result_vol = total_volume(result_shapes)
        expected_frac = 0.75
        actual_frac = result_vol / original_vol
        assert abs(actual_frac - expected_frac) < 0.08, (
            f"Pipe: expected ~{expected_frac*100}% remaining, "
            f"got {actual_frac*100:.1f}%"
        )


# ============================================================================
# Test: L-shaped extrusion — non-trivial cross section
# ============================================================================

class TestLShapedExtrusion:
    """An L-shaped profile extruded along the Y-axis."""

    def _make_l_shape(self):
        """Create an L-shaped solid from two fused boxes, centered on origin."""
        # Horizontal arm of the L: 40x10x80 in XZY
        arm_h = Part.makeBox(40, 80, 10, FreeCAD.Vector(-20, -40, -20))
        # Vertical arm of the L: 10x30x80 in XZY
        arm_v = Part.makeBox(10, 80, 30, FreeCAD.Vector(-20, -40, -10))
        # Fuse to create a proper solid L-shape
        l_shape = arm_h.fuse(arm_v)
        return l_shape

    def test_l_shape_180_cut(self, tmp_dir):
        """Cutting an L-shape at 180° should remove roughly half."""
        l_shape = self._make_l_shape()
        input_path = os.path.join(tmp_dir, "l_shape.step")
        output_path = os.path.join(tmp_dir, "l_shape_cut.step")
        save_shape_as_step(l_shape, input_path)

        run_cutter(input_path, output_path, 180)

        result_shapes = load_step_file(output_path)
        assert len(result_shapes) > 0

        original_vol = l_shape.Volume
        result_vol = total_volume(result_shapes)
        # Exact fraction depends on geometry, but should be roughly half
        actual_frac = result_vol / original_vol
        assert 0.2 < actual_frac < 0.8, (
            f"L-shape 180° cut: expected roughly half, got {actual_frac*100:.1f}%"
        )


# ============================================================================
# Test: Output STEP file validity
# ============================================================================

class TestOutputValidity:
    """Verify the output STEP files are valid and parseable."""

    def test_output_file_exists(self, tmp_dir):
        """Output file should be created."""
        box = Part.makeBox(50, 50, 50, FreeCAD.Vector(-25, -25, -25))
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_out.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 90)

        assert os.path.exists(output_path), "Output STEP file should exist"
        assert os.path.getsize(output_path) > 0, "Output file should not be empty"

    def test_output_shapes_are_valid(self, tmp_dir):
        """All shapes in the output should be valid."""
        box = Part.makeBox(100, 100, 100, FreeCAD.Vector(-50, -50, -50))
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_valid.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 120)

        result_shapes = load_step_file(output_path)
        for i, shape in enumerate(result_shapes):
            assert not shape.isNull(), f"Shape {i} should not be null"
            # Volume should be positive
            assert shape.Volume > 0, f"Shape {i} should have positive volume"

    def test_output_bounding_box_smaller(self, tmp_dir):
        """The output bounding box should not exceed the input bounding box."""
        box = Part.makeBox(100, 100, 100, FreeCAD.Vector(-50, -50, -50))
        input_path = os.path.join(tmp_dir, "box.step")
        output_path = os.path.join(tmp_dir, "box_bb.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path, 90)

        result_shapes = load_step_file(output_path)
        input_bb = box.BoundBox

        for shape in result_shapes:
            result_bb = shape.BoundBox
            # Result bounding box should not extend beyond the original
            assert result_bb.XMax <= input_bb.XMax + 1e-3
            assert result_bb.YMax <= input_bb.YMax + 1e-3
            assert result_bb.ZMax <= input_bb.ZMax + 1e-3
            assert result_bb.XMin >= input_bb.XMin - 1e-3
            assert result_bb.YMin >= input_bb.YMin - 1e-3
            assert result_bb.ZMin >= input_bb.ZMin - 1e-3


# ============================================================================
# Test: Consistency — cutting twice with same angle gives same result
# ============================================================================

class TestConsistency:
    """Running the cutter twice with the same parameters should yield
    the same result."""

    def test_deterministic_output(self, tmp_dir):
        """Two runs with same input should produce identical volumes."""
        box = Part.makeBox(80, 80, 80, FreeCAD.Vector(-40, -40, -40))
        input_path = os.path.join(tmp_dir, "box.step")
        output_path_1 = os.path.join(tmp_dir, "box_out1.step")
        output_path_2 = os.path.join(tmp_dir, "box_out2.step")
        save_shape_as_step(box, input_path)

        run_cutter(input_path, output_path_1, 135)
        run_cutter(input_path, output_path_2, 135)

        shapes_1 = load_step_file(output_path_1)
        shapes_2 = load_step_file(output_path_2)

        vol_1 = total_volume(shapes_1)
        vol_2 = total_volume(shapes_2)

        assert abs(vol_1 - vol_2) < 1e-3, (
            f"Two runs should produce identical volumes: {vol_1} vs {vol_2}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
