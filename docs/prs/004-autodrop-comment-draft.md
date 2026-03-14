Autodrop is now a real backend placement stage instead of a viewport-only animation.

Key points:

- it compares `local_only` against `proxy_then_local`
- the final placement uses only local top/bottom contact bands from the full meshes
- settle offsets persist into later scene/render/export/cut flows

Measured on `outer_1.STEP` + `outer_2.STEP` before the follow-up prep optimization:

- `local_only`: about `2.132s`
- `proxy_then_local`: about `2.115s`

Conclusion:

The rough proxy-drop step only saved about `0.017s` on this geometry, so the meaningful optimization target was contact-mesh preparation rather than the coarse drop itself.
