# CAD Cutter

A Python pipeline for assembling, cutting, and rendering multi-part CAD models.
Takes individual CAD or mesh files, stacks them concentrically by naming
convention, applies configurable radial wedge cuts, and exports the result as
STEP files with optional photorealistic PNG renders.

---

## Features

- **Concentric assembly** — parts named `outer_1`, `mid_1`, `inner_1`, etc.
  are automatically nested and vertically stacked.
- **Radial wedge cutting** — remove an angular sector (e.g. 90°) to expose
  internal structure, with two cutting engines:
  - *Boolean* (`BRepAlgoAPI_Cut`) — robust progressive-tolerance approach.
  - *Direct geometry* (`--cut-direct`) — splits surfaces along cut planes
    and filters solids, producing clean planar cap faces.
- **Segment splitting** — parts suffixed `a` or `b` (e.g. `inner_2a`,
  `inner_2b`) are independently cut into left/right halves of the remaining
  arc.
- **Multi-format input** — STEP, IGES, BREP, STL, OBJ, PLY, and 3MF.
  Mesh files are automatically sewn into solids for boolean compatibility.
- **Material colors** — 25+ material keywords (`copper`, `steel`, `gold`, …)
  detected from filenames.
- **Cylinder orientation** — PCA-based auto-rotation aligns cylindrical parts
  with the Z axis before stacking.
- **Professional rendering** — PyVista off-screen renderer with a five-light
  rig, SSAO, anti-aliasing, and gradient background.  An alternative
  Blender-based renderer (`batch_render.py`) is included for production-quality
  Cycles renders.
- **WRL thermal gradient capability** — includes vendored
  `wrl-color-gradient-app` for thermal diffusion coloring on
  `.wrl/.vrml/.stl/.3mf` meshes to colored `.ply` with optional `.svg` preview.

---

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| [CadQuery](https://cadquery.readthedocs.io/) ≥ 2.4 | CAD kernel (bundles OpenCascade / OCP) |
| NumPy ≥ 1.24 | Numerical operations |
| PyVista ≥ 0.43 | Off-screen rendering |
| Trimesh ≥ 4.0 | Mesh format conversion |
| NetworkX ≥ 3.0 | Trimesh dependency |
| lxml ≥ 4.9 | 3MF support |
| pytest ≥ 7.0 | Test framework |
| `libgl1` (system package) | Provides `libGL.so.1` needed by CadQuery/OCP at import time |

> Linux: install once with `sudo apt-get install -y libgl1` (the included `setup_venv.sh` now attempts this automatically when run as root).

---

## Quick Start

```bash
# Simple vertical stack and export
python assemble.py plate.step shaft.step flange.step -o assy.step

# Concentric assembly with a 90° radial cut and PNG render
python assemble.py outer_1.step mid_1.step inner_1.step outer_2.step \
    -o assembly.step --cut-angle 90 --render assembly.png

# Same cut using the direct geometry engine (faster, cleaner caps)
python assemble.py outer_1.step mid_1.step inner_1.step outer_2.step \
    -o assembly.step --cut-angle 90 --cut-direct --render assembly.png
```

---

## Part Naming Convention

Parts are named as **`<tier>_<level>[segment]`** where:

| Field | Values | Description |
|-------|--------|-------------|
| `tier` | `outer`, `mid`, `inner` | Nesting position (outside → inside) |
| `level` | Integer ≥ 1 | Vertical stack order |
| `segment` | `a` or `b` *(optional)* | Left/right half after cutting |

**Examples:**

| Filename | Tier | Level(s) | Segment |
|----------|------|----------|---------|
| `outer_1.step` | outer | 1 | — |
| `mid_2.step` | mid | 2 | — |
| `inner_3a_copper.step` | inner | 3 | a |
| `outer_1_2.step` | outer | 1–2 (spanning) | — |

Parts without a recognized name are treated as sequential outer parts.

**Material keywords** anywhere in the filename assign colors automatically:
`copper`, `steel`, `aluminum`, `gold`, `brass`, `bronze`, `titanium`,
`wood`, `rubber`, `ceramic`, `glass`, and more.

---


### Web UI

Run the interactive browser UI (default port **12080**):

```bash
python web_ui.py
# then open http://localhost:12080
```

The UI supports:
- selecting parts from the working directory
- side-scrolling 3D thumbnails (orbit mouse controls)
- a main 3D workspace with combined view and tile view
- per-part manual rotation (X/Y/Z) and manual scaling near each thumbnail
- pipeline-stage buttons for auto-orient, auto-scale, cut inner from mid, export parts, render whole, and export whole
- a WRL gradient capability panel for thermal-coloring `.wrl/.vrml/.stl/.3mf` files

## Command-Line Reference

```
python assemble.py <inputs...> [options]
```

### Gradient capability (CLI)

```bash
python assemble.py input.stl --gradient-only \
  --gradient-mode top-bottom \
  --gradient-output colored_output.ply \
  --gradient-render colored_output.svg
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `inputs` | *(required)* | One or more CAD/mesh files or glob patterns |
| `-o`, `--output` | `assembly.step` | Output STEP file path |
| `--axis` | `z` | Stacking axis (`x`, `y`, or `z`) |
| `--gap` | `0` | Gap between stacked levels (model units) |
| `--cut-angle` | *(none)* | Radial wedge angle in degrees (e.g. `90`, `180`) |
| `--cut-direct` | *(off)* | Use direct geometry splitting instead of boolean cut |
| `--render` | *(none)* | Output PNG path for rendering |
| `--resolution` | `2048` | Render resolution in pixels |
| `--cyl` | *(off)* | Auto-orient cylinder axis to Z before stacking |
| `--validate` | *(off)* | Run validation pipeline (3D y=0 cut vs 2D masked top-down projection) |
| `--validate-resolution` | `512` | Resolution used for validation render masks |
| `--validate-max-mismatch` | `0.01` | Allowed normalized pixel mismatch for validation |
| `--parts [DIR]` | *(off)* | Export rotated/scaled pre-stack parts to a directory in native formats (`parts/` by default) |
| `--midscale` | *(off)* | After orient/stack, scale each `mid_*`: uniform XY until its volume contacts matching `outer_*`, then independent Z until it contacts a `mid_*`/`outer_*` above and below. |
| `--mid_cut` | *(off)* | Cut each `mid_*` part to clear matching `inner_*` geometry with 0.02 in clearance; exports cut mid parts to `parts/` (or `--parts` dir) |
| `--debug` | *(off)* | Verbose logging; includes cutter shape in output |
| `--gradient-only` | *(off)* | Run only WRL/STL/3MF thermal gradient capability and skip CAD assembly pipeline |
| `--gradient-output` | `colored_output.ply` | Output PLY path for gradient capability |
| `--gradient-render` | *(none)* | Optional SVG render output for gradient capability |
| `--gradient-mode` | `top-bottom` | Thermal source/sink placement mode (`top-bottom` or `radial`) |

---


## Validation Pipeline

Use `--validate` to run a geometric/image-consistency check useful for tests:

1. Perform a 3D half-space cut at the plane `y=0`.
2. Render the cut result from a top-down camera (`x,y=0,0` looking along +Z).
3. Render the uncut part from the same view, then remove half of the image at `y=0`.
4. Compare binary occupancy masks from both images and fail if mismatch exceeds
   `--validate-max-mismatch`.

```bash
python assemble.py outer_1.step --validate --validate-resolution 512
```


### Fast STL/Mesh Loading Notes

Large mesh-heavy jobs can spend most time in mesh→solid sewing. Inspired by tools like **FSTL**
(which stay triangle-native for interactive speed), `assemble.py` now loads mesh inputs
as lightweight triangle shells by default and defers solid conversion unless explicitly needed.
This significantly reduces startup/load time for many-part STL assemblies while preserving
cutting/rendering behavior.

## Cutting Engines

### Boolean Cut (default)

Uses `BRepAlgoAPI_Cut` with a cylindrical-sector cutter. A progressive
tolerance strategy (five levels from 1e-5 to 1e-1) with topological
validation and volumetric cross-checks ensures robust results even on
complex or mixed CAD/mesh geometry.

```bash
python assemble.py *.step --cut-angle 90 -o cut.step
```

### Direct Geometry Cut (`--cut-direct`)

Splits shapes along the two half-planes that define the cut boundaries using
`BRepAlgoAPI_Splitter`. The splitter re-trims the underlying surface
definitions (cylinders, B-splines, planes, etc.) to new parametric extents
and inserts exact planar faces at the cut surfaces. Resulting solids are
filtered by their angular position to discard the removed wedge.

```bash
python assemble.py *.step --cut-angle 90 --cut-direct -o cut.step
```

**Advantages over boolean cutting:**
- Produces geometrically exact planar cap surfaces
- No progressive tolerance retries needed
- Typically faster on clean CAD geometry

---

## Usage Examples

### Basic Assembly

```bash
# Stack three unnamed parts vertically along Z
python assemble.py plate.step shaft.step flange.step \
    -o stack.step --render stack.png
```

### Concentric Multi-Level Assembly

```bash
# Two levels with inner/mid/outer nesting and a 2 mm gap
python assemble.py outer_1.step mid_1.step inner_1.step \
                    outer_2.step mid_2.step inner_2.step \
    -o concentric.step --gap 2 --render concentric.png
```

### Radial Cut with Segment Splitting

```bash
# 90° wedge cut; inner_2 split into left (a) and right (b) halves
python assemble.py outer_1.step outer_2.step \
                    inner_2a.step inner_2b.step \
    -o segments.step --cut-angle 90 --render segments.png
```

### Cylinder Orientation

```bash
# Auto-detect cylinder axis and align to Z before stacking
python assemble.py barrel.step cap.step --cyl -o barrel_assy.step
```

### Mixed Formats

```bash
# Combine CAD and mesh files
python assemble.py housing.step gasket.stl rotor.obj \
    -o mixed.step --cut-angle 120 --cut-direct
```

### Glob Patterns

```bash
# Case-insensitive glob — finds .step, .STEP, .Step, etc.
python assemble.py parts/*.step -o all.step --cut-angle 90
```

### Debug Mode

```bash
# Verbose output with cutter geometry included in the output STEP
python assemble.py *.step --cut-angle 90 --debug -o debug.step
```

---

## Blender Batch Renderer

For production-quality renders using Blender's Cycles engine:

```bash
blender --background --python batch_render.py
```

Automatically detects materials from part names and applies Principled BSDF
shading, HDRI environment lighting, and noise bump mapping.

---

## FreeCAD Standalone Cutter

A standalone cutter using the FreeCAD API:

```bash
python cut.py input.step output.step 90
python cut.py input.step output.step 90 --debug
```

---

## Testing

```bash
# Run the full test suite (136 tests)
xvfb-run -a python -m pytest test_assemble.py -v

# Run only the direct geometry cut tests
python -m pytest test_assemble.py::TestDirectGeometryCut -v
```

### Test Coverage

| Test Class | Focus |
|------------|-------|
| `TestBoundingBox` | Geometry helpers |
| `TestConcentricStacking` | Tier/level stacking |
| `TestSpanningParts` | Multi-level parts |
| `TestCutting` | Boolean cut operations |
| `TestSegmentCutting` | a/b segment splitting |
| `TestDirectGeometryCut` | Direct geometry splitting |
| `TestOrientToCylinder` | PCA-based orientation |
| `TestMixedFormats` | CAD + mesh integration |
| `TestCutRobustness` | Progressive tolerance strategies |

---

## Project Structure

```
cad_cutter/
  assemble.py        Main assembly pipeline (stack, cut, render)
  cut.py             Standalone FreeCAD-based cutter
  batch_render.py    Blender Cycles batch renderer
  test_assemble.py   Test suite for assemble.py
  test_cut.py        Test suite for cut.py
  requirements.txt   Python dependencies
  *.step / *.STEP    Sample CAD files
```

---

## License

See repository for license details.
