# WRL Color Gradient / Thermal Mesh Colorizer

This project colors mesh faces based on a simple thermal diffusion simulation.

## Supported input formats
- `.wrl` / `.vrml`
- `.stl` (ASCII and binary)
- `.3mf`

## Typical usage
```bash
python wrl_color_gradient.py input.stl -o output.ply --mode top-bottom
```

## Generate requested demo renders
```bash
python wrl_color_gradient.py --generate-test-renders --test-output-dir artifacts
```

This generates:
- cube top-to-bottom thermal coloring
- cylinder radial thermal coloring

Outputs are generated artifacts and are intentionally ignored from git.
