#!/usr/bin/env python3
"""Unit tests for the midlayer topology-optimization adapter scaffolds."""

from __future__ import annotations

from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

import assemble
from topopt_midlayer import MidlayerSectionInput, get_adapter, get_solver_defaults


def _shape_to_mesh(shape) -> trimesh.Trimesh:
    verts, faces = assemble.tessellate_shape(shape, tolerance=0.8)
    return trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=float),
        faces=np.asarray(faces[:, 1:], dtype=np.int64),
        process=False,
    )


def _section_shell(bottom_radius: float = 45.0, top_radius: float = 40.0, height: float = 110.0, wall: float = 6.0):
    outer = cq.Workplane("XY").add(
        cq.Solid.makeCone(
            bottom_radius,
            top_radius,
            height,
            pnt=cq.Vector(0, 0, -height / 2.0),
            dir=cq.Vector(0, 0, 1),
        )
    )
    inner = cq.Workplane("XY").add(
        cq.Solid.makeCone(
            bottom_radius - wall,
            top_radius - wall,
            height,
            pnt=cq.Vector(0, 0, -height / 2.0),
            dir=cq.Vector(0, 0, 1),
        )
    )
    return outer.cut(inner)


def _pill(radius: float = 18.0, height: float = 92.0):
    cyl_height = max(height - (2.0 * radius), 1.0)
    body = cq.Workplane("XY").cylinder(cyl_height, radius)
    top = cq.Workplane("XY").transformed(offset=(0, 0, cyl_height / 2.0)).sphere(radius)
    bottom = cq.Workplane("XY").transformed(offset=(0, 0, -cyl_height / 2.0)).sphere(radius)
    return body.union(top).union(bottom)


def test_midlayer_solver_defaults_expose_both_real_solver_scaffolds():
    defaults = get_solver_defaults()
    assert set(defaults) == {"dl4to", "pymoto"}
    assert defaults["dl4to"]["availability"]["installed"] is False
    assert defaults["dl4to"]["schema"]
    assert defaults["pymoto"]["defaults"]["resolution"] >= 16


def test_midlayer_adapter_generates_mesh_artifacts_for_each_solver(tmp_path: Path):
    outer_mesh = _shape_to_mesh(_section_shell().val().wrapped)
    inner_mesh = _shape_to_mesh(_pill().val().wrapped)
    section = MidlayerSectionInput(
        level=1,
        outer_name="outer_1",
        inner_name="inner_1",
        outer_mesh=outer_mesh,
        inner_mesh=inner_mesh,
    )

    for solver_id in ("dl4to", "pymoto"):
        adapter = get_adapter(solver_id)
        result = adapter.run(
            [section],
            {
                "resolution": 18,
                "volume_fraction": 0.30,
                "smoothing_passes": 2,
            },
            output_dir=tmp_path / solver_id,
        )
        assert result.mode == "scaffold"
        assert result.artifacts
        artifact = result.artifacts[0]
        assert artifact.output_path is not None
        assert artifact.output_path.exists()
        assert len(artifact.mesh.faces) > 0

        outer_bounds = outer_mesh.bounds
        mid_bounds = artifact.mesh.bounds
        assert mid_bounds[0][0] >= outer_bounds[0][0] - 10.0
        assert mid_bounds[0][1] >= outer_bounds[0][1] - 10.0
        assert mid_bounds[1][0] <= outer_bounds[1][0] + 10.0
        assert mid_bounds[1][1] <= outer_bounds[1][1] + 10.0
