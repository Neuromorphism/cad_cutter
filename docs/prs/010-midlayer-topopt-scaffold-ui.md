## Summary

This PR adds a proxy-first midlayer topology-design path to the web UI and
backend, with two solver-facing adapter scaffolds:

- `DL4TO`
- `pyMOTO`

The web UI never operates on exact CAD for this step. It derives a proxy design
volume between matching `outer_*` and `inner_*` section pairs, runs the selected
solver scaffold with configurable defaults, exports generated `mid_*` STL
artifacts, and loads those generated mids back into the current session.

## What Changed

- Added a reusable topology-design adapter package in
  [topopt_midlayer/adapters.py](/home/me/gits/cad_cutter/topopt_midlayer/adapters.py)
  and [topopt_midlayer/__init__.py](/home/me/gits/cad_cutter/topopt_midlayer/__init__.py)
  with:
  - solver registry and availability metadata
  - normalized config handling
  - a stable `MidlayerAdapter` API
  - scaffold implementations for `dl4to` and `pymoto`
  - STL artifact export helpers

- Extended [web_ui.py](/home/me/gits/cad_cutter/web_ui.py) with:
  - `GET /api/midlayer-solvers`
  - config persistence for `midlayer_configs`
  - `design_midlayer_dl4to` and `design_midlayer_pymoto` stage handlers
  - matching-section validation for loaded `outer_*` / `inner_*` pairs
  - generated-part insertion back into the active session

- Added a new Midlayer Design section to
  [templates/index.html](/home/me/gits/cad_cutter/templates/index.html) with:
  - `Design midlayer (DL4TO)`
  - `Design midlayer (pyMOTO)`
  - a collapsible configuration panel

- Added frontend config rendering and stage handling in
  [static/app.js](/home/me/gits/cad_cutter/static/app.js), including longer
  watchdog thresholds for the midlayer design stages.

- Added styling for the new config UI in
  [static/style.css](/home/me/gits/cad_cutter/static/style.css).

- Added backend and browser regression coverage in:
  - [test_topopt_midlayer.py](/home/me/gits/cad_cutter/test_topopt_midlayer.py)
  - [test_web_ui.py](/home/me/gits/cad_cutter/test_web_ui.py)
  - [tests/playwright/webui.spec.js](/home/me/gits/cad_cutter/tests/playwright/webui.spec.js)

- Added a solver survey note in
  [docs/topopt-solver-survey.md](/home/me/gits/cad_cutter/docs/topopt-solver-survey.md)
  covering real solver candidates and the architecture tradeoffs.

## Current Solver Mode

The adapter boundary is real, but in this environment both solvers are running
in scaffold mode because neither native package is installed:

- `dl4to`
- `pymoto`

The UI exposes that availability explicitly, and the generated outputs are
still deterministic organic support meshes intended for authentic test-model
generation.

## Validation

Python validation:

```bash
pytest -q test_topopt_midlayer.py test_web_ui.py
```

Result:

- `31 passed`

Browser coverage added:

```bash
npx playwright test tests/playwright/webui.spec.js -g "designs midlayers through both solver buttons without hanging"
```

That browser test was added but not executed in this sandbox because Chromium
launch is blocked by the sandbox runtime.

## Sample Outputs

Generated example artifacts:

- [.webui_cache/midlayer_demo/dl4to/mid_1_dl4to.stl](/home/me/gits/cad_cutter/.webui_cache/midlayer_demo/dl4to/mid_1_dl4to.stl)
- [.webui_cache/midlayer_demo/pymoto/mid_1_pymoto.stl](/home/me/gits/cad_cutter/.webui_cache/midlayer_demo/pymoto/mid_1_pymoto.stl)

Example scaffold output characteristics from `outer_1.step` + `inner_1.step`:

- `mid_1_dl4to.stl`: `30300` faces, `4413` vertices
- `mid_1_pymoto.stl`: `140424` faces, `13754` vertices
