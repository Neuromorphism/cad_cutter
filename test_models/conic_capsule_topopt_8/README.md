This directory contains a generated eight-section test assembly for `assemble.py`.

Model summary:
- `outer_1.step` through `outer_8.step`: conic outer shell sections, each a frustum.
- `inner_1.step` through `inner_6.step`: pill-shaped inner bodies.
- `mid_1.step` through `mid_6.step`: organic suspension structures intended to resemble topology-optimized supports.

Requested dimensions:
- Total height: `4000 mm`
- Section count: `8`
- Section height: `500 mm`
- Outer base diameter at section 1: `500 mm`
- Inner diameter at section 1: `200 mm`
- Sections `1` through `6` include inner and mid parts
- Sections `7` and `8` are outer-only

Regenerate the files with:

```bash
python test_models/conic_capsule_topopt_8/generate.py
```

Assemble them with:

```bash
python assemble.py test_models/conic_capsule_topopt_8/*.step -o test_models/conic_capsule_topopt_8/assembly.step --render test_models/conic_capsule_topopt_8/assembly.png
```
