## Summary

This PR hardens the browser workflow against hangs by keeping the pipeline on cached preview meshes and adding end-to-end browser performance budgets for the real heavy datasets that previously wedged the UI.

## What Changed

- keeps `auto-orient` on a fast axis-aligned preview path by default
- prevents stage completion from falling back to slow exact-shape preview rebuilds
- keeps `auto-stack`, `auto-scale`, and `autodrop` on preview-mesh bounds/contact calculations for browser workflows
- speeds up `export parts`, `render assembly`, and `export assembly` for heavy mesh sessions by using transformed preview meshes instead of exact CAD regeneration where exact topology is not needed
- adds a Playwright stage-budget regression that fails on any `stall-warning`, `api-error`, `scene-error`, or `load-error`
- adds a three-cycle hang regression:
  - load STEP pair and run pipeline
  - reload and load STL pair and run pipeline
  - reload and load a third cylinder-model set and run pipeline

## Browser Timing Results

Measured from the new Playwright regression:

| Dataset | Auto-orient | Auto-stack | Auto-scale | Autodrop | Export parts | Render assembly | Export assembly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `outer_1.STEP + outer_2.STEP` | 1.111s | 0.786s | 0.942s | 3.716s | 1.773s | 1.918s | 1.582s |
| `out.stl + out2.stl` | 3.276s | 3.256s | 3.287s | 3.485s | 3.381s | 3.076s | 3.106s |

All measured stages are under the 5 second budget.

## Hang Regression Evidence

The sequential reload test captures proof that the browser reached later datasets after completing earlier workflows:

- second dataset screenshot: `.webui_cache/playwright_artifacts/second-set-loaded.png`
- third dataset screenshot: `.webui_cache/playwright_artifacts/third-set-loaded.png`
- timing artifact: `.webui_cache/playwright_artifacts/pipeline_timings.json`

The third dataset also covers the `Cut inner from mid` stage on a realistic cylinder workflow:

- `test_models/conic_capsule_topopt_8/outer_1.step`
- `test_models/conic_capsule_topopt_8/mid_1.step`
- `test_models/conic_capsule_topopt_8/inner_1.step`

## Validation

```bash
pytest -q test_web_ui.py
npx playwright test tests/playwright/pipeline_performance.spec.js
```
