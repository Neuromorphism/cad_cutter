# PR 002: STEP Preview Engine Research Prototype

## Summary

This PR adds a research/prototype track for a future high-performance STEP
preview engine.

## Includes

1. `brep_engine.step_index`: a fast textual STEP scanner that reports schema,
   top entity counts, and whether tessellated STEP entities are present.
2. `brep_engine.preview_benchmark`: a benchmark harness for the current preview
   path.
3. `docs/step-preview-engine.md`: architecture notes and current limits.
4. `test_step_index.py`: regression coverage for the scanner.

## Key finding

`outer_1.STEP` is an exact AP214 BRep file with entities such as
`ADVANCED_FACE`, `B_SPLINE_CURVE_WITH_KNOTS`, and `CYLINDRICAL_SURFACE`.
The scanner does not find tessellated entities. That means a true
from-scratch sub-second engine would need a dedicated parser and triangulator,
not just a thin wrapper around existing exact-kernel imports.

## Current benchmark

`python -m brep_engine.preview_benchmark outer_1.STEP` reports a warmed first
scene of about `0.08s` and a repeat scene of about `0.02s` in this repo.

## Validation

1. `pytest -q test_step_index.py`
2. `python -m brep_engine.step_index outer_1.STEP`
3. `python -m brep_engine.preview_benchmark outer_1.STEP`

## Reviewer focus

1. Confirm the research notes are technically accurate.
2. Decide whether to treat this as a prototype branch or merge only the docs
   and scanner while a native implementation is designed separately.
