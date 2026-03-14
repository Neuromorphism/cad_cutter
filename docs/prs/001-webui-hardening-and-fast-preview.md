# PR 001: Web UI Hardening And Fast Preview

## Summary

This PR hardens the CAD Cutter web UI and makes large-model preview loading
substantially faster.

## Includes

1. Directory browser overlay and working `Change Dir` flow.
2. Fixed header tooltip clipping with a real floating tooltip element.
3. Local Three.js vendor assets instead of CDN dependencies.
4. Playwright browser coverage for the web UI shell, directory navigation,
   and part loading.
5. Atomic `/api/load` behavior so invalid model files do not wipe the session.
6. Richer progress reporting with current-action text, recent activity, and
   elapsed-time badges.
7. Large-CAD preview surrogates using sibling mesh files where available.
8. Persistent preview mesh-payload caching.
9. Direct mesh-file preview loading for STL/OBJ/PLY/3MF sources instead of
   re-tessellating those files through OCP.
10. Combined-scene transform reuse so the server can reference already-built
    part meshes instead of rebuilding moved copies.

## Validation

1. `pytest -q test_web_ui.py`
2. `npm run test:webui`

## Reviewer focus

1. Verify the preview-only upgrade path is acceptable for topology-sensitive
   operations.
2. Confirm the on-disk preview cache location and format are acceptable.
3. Check UI language and visibility of the new preview badge.
