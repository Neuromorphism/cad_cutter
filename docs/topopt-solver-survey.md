# Topology Optimization Solver Survey

This repo currently has:

- `numpy`
- `scipy`
- `trimesh`
- `cadquery`
- `pyvista`

It does not currently have:

- `torch`
- `dl4to`
- `pymoto`
- `topopt`
- `topy`

## DL4TO

Primary sources:

- https://github.com/dl4to/dl4to
- https://dl4to.github.io/dl4to/

Observed fit:

- Real 3D topology optimization library
- PyTorch-based
- Structured-grid / voxel-first workflow
- Includes its own linear-elasticity PDE solver and SIMP-based optimization
- Research-oriented and strong for learned / differentiable workflows

Risks for this repo:

- Requires `torch`, which is not present
- Recommends `pyvista==0.38.1`, while this repo already uses a much newer `pyvista`
- Best fit when we are willing to build a voxel/FDM pipeline around the optimizer

## pyMOTO

Primary sources:

- https://pymoto.readthedocs.io/
- https://pypi.org/project/pymoto/

Observed fit:

- Real topology optimization framework
- Supports 2D and 3D problems
- Includes OC, MMA, and GCMMA optimizers
- Depends mainly on `numpy` and `scipy`, with optional faster sparse solvers
- Modular formulation is a good fit for experimentation

Why it is attractive here:

- Closest match to the repo's existing scientific Python stack
- Lower dependency jump than DL4TO
- Better fit if we want a real solver without committing to a PyTorch stack

## Scikit-Topt

Primary sources:

- https://pypi.org/project/scikit-topt/
- https://joss.theoj.org/papers/10.21105/joss.09092

Observed fit:

- Recent Python topology optimization library
- Built on `scipy` and `scikit-fem`
- Supports OC-style density-method workflows
- Emphasizes algorithm development and sparse-matrix performance

Risks for this repo:

- Would add `scikit-fem`, `pyamg`, `numba`, and `meshio`
- Strong candidate, but the dependency footprint is larger than pyMOTO

## TopOpt

Primary source:

- https://github.com/zfergus/topopt

Observed fit:

- Promising modern Python library
- Has 3D regular-grid and 3D general-mesh targets in scope
- Explicitly marked as early-stage in its README

Assessment:

- Worth watching
- Less stable choice than pyMOTO or DL4TO for immediate integration

## ToPy

Primary source:

- https://github.com/williamhunter/topy

Observed fit:

- Longstanding topology optimization project
- Supports 2D and 3D
- README notes the stable release is Python 2-era and Python 3 support is effectively the newer unstable path

Assessment:

- Historically important
- Not a strong default choice for a modern Python 3.12 repo

## Recommendation

If this repo wants a *real external solver* soon:

1. `pyMOTO` is the most practical first integration target.
2. `DL4TO` is the most interesting research-grade 3D solver, but only if we are willing to adopt a PyTorch + voxel/FDM path.
3. `Scikit-Topt` is a serious alternative if we want a more classical finite-element Python stack and accept the larger dependency set.

For this repo specifically:

- The current mesh/proxy generator path is still useful for fixture generation.
- The cleanest upgrade path is:
  1. keep the current in-repo generator for deterministic fixtures
  2. add an optional external-solver adapter layer
  3. start that adapter with `pyMOTO` or `DL4TO`, depending on whether we want lower dependency cost or deeper 3D research capability
