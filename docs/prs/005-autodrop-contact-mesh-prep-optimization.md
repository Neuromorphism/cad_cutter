## Summary

This PR reduces `Autodrop` time by bypassing exact-shape tessellation during contact-mesh preparation when cached mesh payloads are already available.

## What Changed

- added `assemble.payload_to_clean_trimesh(...)` to build cleaned trimesh objects directly from cached payload data
- updated the fast autodrop solver to accept optional cached mesh payloads
- updated the web UI autodrop stage to reuse cached mesh payloads for untransformed mesh-backed parts instead of rebuilding contact meshes from exact CAD/BRep tessellation
- kept the local-contact refinement logic unchanged:
  - optional proxy coarse drop
  - final flush translation from local top/bottom contact bands on the full meshes
- left the comparison between:
  - `local_only`
  - `proxy_then_local`
  in place so timings still come back in the API response

## Measured Result

On `outer_1.STEP` + `outer_2.STEP`:

- before:
  - `local_only`: about `2.132s`
  - `proxy_then_local`: about `2.115s`
- after:
  - `local_only`: about `0.729s`
  - `proxy_then_local`: about `0.728s`

The main gain came from faster contact-mesh preparation, not from the coarse proxy-drop itself.

## Validation

- `python -m py_compile web_ui.py assemble.py`
- `pytest -q test_web_ui.py`
- `npm run test:webui`
