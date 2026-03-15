## Summary

This PR adds a dedicated repo-root pipeline regression harness and fixes the
`Auto-stack` ordering logic for repo-root `outer_1.STEP` + `outer_2.STEP`.

It does three things:

1. Fixes `Auto-stack` for the `cylinder` workflow so parseable parts are sorted
   by level and tier instead of raw centroid order.
2. Adds backend stage regressions for the repo-root STEP pair and a compatible
   cut-stage trio.
3. Adds a dedicated browser regression suite and Playwright config for the
   repo-root pipeline flow, isolated on its own port so it does not collide
   with the main browser harness.

## Key Changes

- `web_ui.py`
  - add `_auto_stack_sort_key(...)`
  - make `auto_stack` prefer `(level, tier, segment)` ordering in the
    `cylinder` workflow
- `test_web_ui.py`
  - add repo-root STEP pair `auto_stack` correctness assertion
  - add shared backend stage-process checks for:
    - `auto_orient`
    - `auto_stack`
    - `auto_scale`
    - `auto_drop`
    - `export_parts`
    - `export_whole`
  - add `cut_inner_from_mid` stage coverage on
    `outer_1.step` / `mid_1.step` / `inner_1.step`
- `tests/playwright/repo_root_pipeline.spec.js`
  - add repo-root browser pipeline checks for:
    - `outer_1.STEP` + `outer_2.STEP` load + `Auto-stack`
    - repo-root STEP stage sequence
    - compatible cut-stage trio
- `playwright.repo-root.config.js`
  - dedicated browser config on `127.0.0.1:12084`
  - isolated server bootstrap for the repo-root suite
- `package.json`
  - add `npm run test:webui:repo-root`

## Validation

Executed locally:

```bash
pytest -q test_web_ui.py
```

Results:
- `pytest -q test_web_ui.py` -> `22 passed`

Browser suite:
- `npm run test:webui:repo-root`
- user-reported result: `3 passed (32.8s)`

## Notes

- The repo-root browser suite now self-hosts on port `12084`, so it does not
  depend on a manually running web UI server and does not contend with the main
  Playwright harness on `12082`.
