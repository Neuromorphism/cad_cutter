## Summary

This PR hardens the web UI load path for large STL models and makes the failure/debugging path visible in the browser.

Key changes:
- add a visible web UI version badge and `/api/version`
- add client and server debug logging with copy/export support
- persist client debug events to `/api/debug-log`
- return scene payloads inline from `/api/load` to avoid a fragile follow-up `/api/scene` request on the main load path
- add a fast server-side mesh scene path for large mesh-only loads
- reuse client-side `BufferGeometry` objects instead of rebuilding the same mesh repeatedly
- render thumbnail cards immediately while deferring or skipping expensive 3D previews for very large meshes
- add browser regression coverage for the exact `out.stl` + `out2.stl` case

## Why

The user was seeing the web UI hang while loading `out.stl` and `out2.stl`. Investigation showed multiple contributing issues:
- the old UI could stall between `/api/load` and `/api/scene`
- the browser had poor visibility into where it was stuck
- large mesh loads triggered redundant client-side geometry work
- the previous browser suite only covered small fixture models

This PR addresses all of those directly rather than relying on timeout-only mitigation.

## Implementation Notes

Backend:
- `web_ui.py` now computes and exposes a version string
- debug entries are mirrored to an in-memory buffer and `.webui_cache/debug_log.jsonl`
- `/api/load` accepts `include_scene: true` and can return the scene inline
- mesh-only large STL loads use a fast preview stacking path based on cached mesh bounds
- fast-path scene completion now emits `scene-done` like the standard path

Frontend:
- `static/app.js` includes a collapsible debug panel and `Copy Debug Log`
- API requests include better lifecycle logging and timeout reporting
- heavy geometry now uses cached Three.js `BufferGeometry`
- thumbnail cards appear immediately, but very large meshes do not force synchronous 3D thumbnail rendering
- stalled loads now report useful context instead of appearing silent

Tests:
- `test_web_ui.py` covers the debug log endpoint
- Playwright now verifies:
  - version badge visibility
  - debug log population
  - heavy STL load from repo root
  - no fallback `GET /api/scene` on the main `out.stl` + `out2.stl` load path

## Validation

Executed locally:

```bash
pytest -q test_web_ui.py
npm run test:webui
```

Playwright now passes the heavy STL regression case in the browser.
