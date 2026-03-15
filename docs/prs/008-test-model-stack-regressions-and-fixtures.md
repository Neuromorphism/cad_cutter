## Summary

This PR adds reproducible regression coverage for the generated conic capsule test models and fixes the browser viewport path so large stacked assemblies and more complex repo-root STEP models remain visible and responsive in the Web UI.

It does three things:

1. Makes the `test_models/conic_capsule_topopt_8` fixture set part of the repo so backend and Playwright tests can run on real `outer_*`, `mid_*`, and `inner_*` geometry.
2. Fixes the large-assembly browser display path by keeping the main viewport stable:
   - render thumbnails as static snapshots instead of live WebGL scenes
   - skip 3D thumbnails for large assemblies
   - scale scene fog to the current scene size so the 4 m cone does not fade into the background
3. Adds backend and browser regressions that prove:
   - upload autoload completes without timing out
   - two `outer_*` sections auto-stack vertically
   - the full `outer_*`, `mid_*`, `inner_*` set auto-stacks into the expected cone/nested assembly
   - the browser viewport actually shows geometry, not just successful API calls
4. Codifies the proxy-first viewport rule:
   - the browser works on proxy meshes plus transforms
   - exact CAD stays server-side for export, render, and geometry-changing backend stages
   - small complex proxy scenes can inline proxy meshes to avoid fragile extra HTTP hops
   - large scenes still use deferred proxy fetches

## Key Changes

- `web_ui.py`
  - add deferred `/api/mesh-payload` references for CAD preview meshes
  - reuse fast preview scene payloads for the generated test models
  - keep `auto_orient` on transformed proxy meshes instead of re-tessellating exact CAD after runtime transforms
  - inline small proxy scenes such as `outer_1.STEP` + `outer_2.STEP` directly in the initial load payload
  - add server debug markers around proxy payload generation
- `static/app.js`
  - remote payload fetch for CAD previews
  - static thumbnail snapshot rendering
  - large-assembly thumbnail protection
  - scene-size-aware fog
  - shared `loadSelectedParts()` path for upload autoload and manual load
- `static/style.css`
  - thumbnail image styling for snapshot previews
- `test_web_ui.py`
  - backend assertions for:
    - two outer sections stacking vertically
    - full outer/mid/inner nesting
    - full-load deferred preview payload behavior
    - repo-root `outer_1.STEP` + `outer_2.STEP` inline proxy payload behavior
    - transformed orientation proxies staying on cached preview meshes
- `tests/playwright/test_models_stack.spec.js`
  - browser tests on real generated models
  - screenshot-content assertion to fail on blank viewports
- `tests/playwright/webui.spec.js`
  - repo-root `outer_1.STEP` + `outer_2.STEP` browser regression now asserts visible viewport output, not just error absence
- `docs/display-proxy-architecture.md`
  - documents the core architectural rule for the Web UI
- `test_models/conic_capsule_topopt_8`
  - commit the generated outer/mid/inner fixture set plus generator/readme

## Validation

Executed locally:

```bash
pytest -q test_web_ui.py
npx playwright test tests/playwright/test_models_stack.spec.js
```

Results:
- `pytest -q test_web_ui.py` -> `12 passed`
- `npx playwright test tests/playwright/test_models_stack.spec.js` -> `3 passed`
- targeted backend follow-up after the proxy-first STEP fix:
  - `pytest -q test_web_ui.py -k 'orientation_payload_reuses_preview_mesh_after_runtime_transform or repo_outer_step_pair_uses_inline_proxy_meshes or test_models_full_load_uses_deferred_mesh_payloads'` -> `3 passed`
- direct browser verification against a clean `--no-reload` server on `outer_1.STEP` + `outer_2.STEP`:
  - viewport screenshot change ratio `0.106`
  - no deferred `/api/mesh-payload?path=outer_1.STEP` fetch
  - `Auto-orient complete using axis-aligned six-way heuristic`

## Proof Artifacts

- Browser autoload: `.webui_cache/playwright_artifacts/test-model-autoload.png`
- Browser two-outer stack: `.webui_cache/playwright_artifacts/test-model-two-outer-ui.png`
- Browser full stacked assembly: `.webui_cache/playwright_artifacts/test-model-full-stack-ui.png`
- Backend render of full cone: `.webui_cache/playwright_artifacts/test-model-full-stack-render.png`
- Repo-root complex STEP pair visible in browser: `.webui_cache/playwright_artifacts/manual-root-outer-step.png`

## Notes

- I intentionally did not include unrelated local artifacts such as cutaway renders, `.webui_cache/`, `test-results/`, or other generated outputs.
- The new browser regression depends on the committed `test_models/conic_capsule_topopt_8` fixture set so it can run on a clean checkout.
