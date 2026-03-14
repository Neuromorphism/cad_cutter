## Summary

This PR hardens the web UI against stage hangs and adds browser regressions that explicitly fail when a pipeline step stalls, leaves the loader up, or records client/server error signals.

## What Changed

- kept STEP preview refreshes on cached mesh data after `Auto-orient` and other runtime transforms, instead of forcing a slow exact re-tessellation path
- serialized orientation steps in the scene payload and applied them client-side for combined view, tile view, and thumbnails
- added explicit client-side timeouts for browser-triggered pipeline stages
- added stage-level stall watchdog coverage in the browser
- added `DELETE /api/debug-log` so Playwright can isolate each stage run
- fixed `Export parts` in the web UI to pass the structure expected by `assemble.export_transformed_parts(...)`
- exposed the fast axis-aligned / legacy fine-grained orientation choice in the UI

## Browser No-Hang Coverage

The Playwright suite now verifies that stage execution does not hang by:

- clearing the debug log before each stage
- clicking the stage button through the browser
- waiting for the button to re-enable
- waiting for the viewport loader to disappear
- requiring a completion-like status message
- failing on any of:
  - `stall-warning`
  - `api-error`
  - `scene-error`
  - `load-error`

Covered browser stages:

- `Auto-orient parts`
- `Auto-stack parts`
- `Auto-scale parts`
- `Autodrop`
- `Cut inner from mid`
- `Export parts`
- `Render assembly`
- `Export assembly`

## Validation

- `pytest -q test_web_ui.py`
- `npm run test:webui`
