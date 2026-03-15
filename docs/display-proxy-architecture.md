# Display Proxy Architecture

The web UI is proxy-first.

That means the browser never operates on exact CAD/BRep geometry directly. The
browser only works with:

- display mesh proxies
- object transforms
- lightweight scene metadata

Exact geometry remains server-side and is only used when the user explicitly
requests backend work such as:

- export
- render
- cut / boolean stages
- any exact geometry transform that must be committed to source parts

## Core Rule

Treat display mesh + object transform as separate layers.

- Display layer:
  - cached mesh payloads
  - deferred mesh fetches
  - decimated meshes for thumbnails, orientation, and interaction
  - client-side transform application
- Exact layer:
  - source STEP / BRep / STL import
  - CadQuery / OCP operations
  - export of transformed source parts
  - final renders and exact downstream processing

## Why

This keeps the web UI responsive even when the source geometry is large or
topologically expensive.

If the browser path accidentally falls back to exact geometry, the user sees:

- long load times
- hung pipeline stages
- blank or delayed view updates
- oversized JSON scene payloads

## Consequences For Implementation

- `/api/load` and `/api/scene` should prefer remote proxy mesh references over
  inline exact tessellation.
- Small proxy scenes may inline their proxy meshes directly in the scene payload
  to avoid an extra HTTP hop. Large scenes should stay on deferred mesh fetches.
- Orientation, stacking previews, thumbnails, and viewport updates should stay
  in proxy-mesh space.
- Runtime transforms should be applied to proxy meshes for interaction, then
  replayed onto exact geometry only when a backend stage needs it.
- Re-tessellation of exact shapes is a backend concern, not a viewport concern.

## Regression Expectation

Tests should prove all of the following:

- complex STEP pairs load through deferred proxy payloads
- the viewport visibly shows geometry, not just successful HTTP responses
- UI stages do not hang when operating on proxy meshes
- exact backend exports/renders still reflect the transforms discovered in the
  UI
