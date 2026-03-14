## Summary

This PR improves the web UI workflow for interactive model setup and makes orientation responsive on preview geometry instead of exact CAD.

## What Changed

- increased default viewport lighting so STEP shells and frustums read clearly in both the main view and thumbnails
- added a `Workflow` dropdown with:
  - `Cylinder`
  - `Generic Stack`
- split the old combined orientation/stacking behavior into separate pipeline actions:
  - `Auto-orient parts`
  - `Auto-stack parts`
  - `Auto-scale parts`
  - `Autodrop`
- moved the old physics simulation trigger out of the view toolbar and into the pipeline as `Autodrop`
- added decimated mesh proxies for:
  - thumbnail rendering
  - fast auto-orient solves
- changed `auto_orient` to solve on cached decimated preview meshes, then store the resulting orientation as runtime transform steps so later exact-CAD export/cut stages still honor the orientation
- kept exact geometry off the critical path for `auto_orient` and `auto_stack`
- added a server regression test for the decimated preview auto-orient path
- updated Playwright coverage for the new workflow controls

## Why

The previous `auto_orient` path forced exact geometry and reused the older `orient_to_cylinder` behavior, which was much slower than necessary for interactive UI use. The preview scene already creates lightweight tessellated representations, so this change reuses those cached proxies for fast orientation work while preserving correct exact-geometry behavior for later export/cut stages.

## Validation

- `pytest -q test_web_ui.py`
- `npm run test:webui`
- direct Flask-client timing check:
  - `outer_1.STEP` + `outer_2.STEP`
  - `auto_orient` completed in about `0.02s` on `cylinder` workflow using decimated preview meshes
