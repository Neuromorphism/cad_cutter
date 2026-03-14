# STEP Preview Engine Research

## Current finding

`outer_1.STEP` is an AP214 exact BRep file. It contains `ADVANCED_FACE`,
`B_SPLINE_CURVE_WITH_KNOTS`, `EDGE_CURVE`, and other exact-surface entities.
The scanner in `brep_engine/step_index.py` does not find AP242 tessellated
entities such as `TESSELLATED_*` or `TRIANGULATED_*`.

That matters because a true sub-second "from scratch" engine has two very
different cases:

1. STEP already carries tessellated presentation data:
   a lightweight parser can map triangles straight into the viewport.
2. STEP only carries exact BRep:
   sub-second first frame requires a native parser and triangulator optimized
   for incremental first-frame output, likely in C++ or Rust rather than pure
   Python.

## What this repo now has

1. A fast textual scanner for STEP structure:
   `python -m brep_engine.step_index outer_1.STEP`
2. A repeatable benchmark harness for the existing web preview path:
   `python -m brep_engine.preview_benchmark outer_1.STEP`
3. A warmed preview-cache path in the web UI that can already deliver a
   sub-second first scene on `outer_1.STEP` in this repo.

## Recommended architecture for a real engine

1. Parser:
   a streaming STEP parser that indexes entity offsets without building the
   full object graph up front.
2. Topology graph:
   compact edge/loop/face tables with lazy curve and surface evaluation.
3. Triangulation:
   a dedicated tessellator that emits coarse first-frame triangles before a
   higher-accuracy refinement pass.
4. Cache:
   persistent binary preview payloads keyed by file path, size, and mtime.
5. View path:
   preview payloads should bypass exact-kernel import entirely until the user
   requests topology-sensitive operations.

## Honest status

This branch is a research/prototype track, not a replacement CAD kernel.
It gives the repo tools to distinguish tessellated-vs-exact STEP inputs and to
benchmark the warmed preview path while a real parser/triangulator effort is
being designed.
