"""Topology-optimization adapter scaffolds for midlayer generation."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage
import trimesh
from trimesh.transformations import scale_and_translate

try:
    import pymoto as _pymoto_native
except Exception:
    _pymoto_native = None


DEFAULT_CONFIG: dict[str, Any] = {
    "volume_fraction": 0.25,
    "resolution": 28,
    "smoothing_passes": 8,
    "shell_thickness": 1.35,
    "bridge_count": 6,
    "axial_slices": 4,
    "dl4to_iterations": 48,
    "dl4to_penalty": 3.0,
    "pymoto_iterations": 36,
    "pymoto_penalty": 2.4,
}

SOLVER_REGISTRY: dict[str, dict[str, str]] = {
    "dl4to": {
        "label": "DL4TO",
        "package": "dl4to",
        "summary": "PyTorch voxel topology optimization",
    },
    "pymoto": {
        "label": "pyMOTO",
        "package": "pymoto",
        "summary": "NumPy/SciPy modular topology optimization",
    },
}

_COMMON_SCHEMA: list[dict[str, Any]] = [
    {"key": "volume_fraction", "label": "Keep fraction", "type": "number", "step": 0.05, "min": 0.08, "max": 0.85},
    {"key": "resolution", "label": "Grid resolution", "type": "number", "step": 1, "min": 16, "max": 56},
    {"key": "smoothing_passes", "label": "Smoothing passes", "type": "number", "step": 1, "min": 0, "max": 20},
    {"key": "shell_thickness", "label": "Shell bias", "type": "number", "step": 0.05, "min": 0.25, "max": 4.0},
    {"key": "bridge_count", "label": "Bridge count", "type": "number", "step": 1, "min": 3, "max": 12},
    {"key": "axial_slices", "label": "Axial slices", "type": "number", "step": 1, "min": 2, "max": 8},
]


def _solver_mode(solver_id: str, installed: bool) -> tuple[str, str]:
    if solver_id == "pymoto" and installed:
        return "native", "native package 'pymoto' detected; using native pyMOTO optimization"
    if installed:
        return "scaffold", f"native package '{SOLVER_REGISTRY[solver_id]['package']}' detected; scaffold adapter remains active"
    return "scaffold", f"native package '{SOLVER_REGISTRY[solver_id]['package']}' unavailable; using local scaffold"


@dataclass
class SectionSpec:
    level: int
    name: str
    outer_meshes: list[trimesh.Trimesh]
    inner_meshes: list[trimesh.Trimesh]


@dataclass
class SolverAvailability:
    installed: bool
    package: str
    mode: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MidlayerDesignConfig:
    volume_fraction: float = float(DEFAULT_CONFIG["volume_fraction"])
    resolution: int = int(DEFAULT_CONFIG["resolution"])
    smoothing_passes: int = int(DEFAULT_CONFIG["smoothing_passes"])
    shell_thickness: float = float(DEFAULT_CONFIG["shell_thickness"])
    bridge_count: int = int(DEFAULT_CONFIG["bridge_count"])
    axial_slices: int = int(DEFAULT_CONFIG["axial_slices"])
    dl4to_iterations: int = int(DEFAULT_CONFIG["dl4to_iterations"])
    dl4to_penalty: float = float(DEFAULT_CONFIG["dl4to_penalty"])
    pymoto_iterations: int = int(DEFAULT_CONFIG["pymoto_iterations"])
    pymoto_penalty: float = float(DEFAULT_CONFIG["pymoto_penalty"])
    blur_sigma: float = 0.45
    branch_count: int = int(DEFAULT_CONFIG["bridge_count"])
    shell_bias: float = 1.0
    bridge_bias: float = 1.0
    axial_bias: float = 1.0
    twist_turns: float = 0.5
    ring_bias: float = 1.0
    seed: int = 0

    def normalized(self) -> "MidlayerDesignConfig":
        return MidlayerDesignConfig(
            volume_fraction=float(np.clip(float(self.volume_fraction), 0.08, 0.85)),
            resolution=int(np.clip(int(self.resolution), 16, 56)),
            smoothing_passes=int(np.clip(int(self.smoothing_passes), 0, 20)),
            shell_thickness=float(np.clip(float(self.shell_thickness), 0.25, 4.0)),
            bridge_count=int(np.clip(int(self.bridge_count), 3, 12)),
            axial_slices=int(np.clip(int(self.axial_slices), 2, 8)),
            dl4to_iterations=int(np.clip(int(self.dl4to_iterations), 8, 200)),
            dl4to_penalty=float(np.clip(float(self.dl4to_penalty), 1.0, 6.0)),
            pymoto_iterations=int(np.clip(int(self.pymoto_iterations), 8, 200)),
            pymoto_penalty=float(np.clip(float(self.pymoto_penalty), 1.0, 6.0)),
            blur_sigma=float(np.clip(float(self.blur_sigma), 0.1, 3.0)),
            branch_count=int(np.clip(int(self.branch_count), 3, 12)),
            shell_bias=float(np.clip(float(self.shell_bias), 0.2, 1.5)),
            bridge_bias=float(np.clip(float(self.bridge_bias), 0.0, 1.5)),
            axial_bias=float(np.clip(float(self.axial_bias), 0.2, 1.5)),
            twist_turns=float(np.clip(float(self.twist_turns), 0.0, 3.0)),
            ring_bias=float(np.clip(float(self.ring_bias), 0.0, 1.5)),
            seed=int(self.seed),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())


@dataclass
class MidlayerSectionInput:
    level: int
    outer_name: str
    inner_name: str
    outer_mesh: trimesh.Trimesh
    inner_mesh: trimesh.Trimesh
    axis: str = "z"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MidlayerArtifact:
    level: int
    name: str
    mesh: trimesh.Trimesh
    output_path: Path | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "name": self.name,
            "outputPath": str(self.output_path) if self.output_path is not None else None,
            "metadata": dict(self.metadata),
        }


@dataclass
class MidlayerDesignResult:
    solver_id: str
    label: str
    mode: str
    availability: SolverAvailability
    config: MidlayerDesignConfig
    artifacts: list[MidlayerArtifact]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "solverId": self.solver_id,
            "label": self.label,
            "mode": self.mode,
            "availability": self.availability.to_dict(),
            "config": self.config.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "notes": self.notes,
        }


def solver_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for solver_id, info in SOLVER_REGISTRY.items():
        package = info["package"]
        installed = importlib.util.find_spec(package) is not None
        mode, notes = _solver_mode(solver_id, installed)
        status[solver_id] = {
            "id": solver_id,
            "label": info["label"],
            "summary": info["summary"],
            "package": package,
            "native_package": installed,
            "availability": {
                "installed": installed,
                "package": package,
                "mode": mode,
                "notes": notes,
            },
            "mode": mode,
            "detail": notes,
        }
    return status


def coerce_config(config: dict[str, Any] | MidlayerDesignConfig | None) -> dict[str, Any]:
    if isinstance(config, MidlayerDesignConfig):
        normalized = config.normalized()
    else:
        merged = MidlayerDesignConfig()
        if isinstance(config, dict):
            for key, value in config.items():
                if hasattr(merged, key):
                    setattr(merged, key, value)
        normalized = merged.normalized()
    return normalized.to_dict()


def get_solver_defaults() -> dict[str, dict[str, Any]]:
    status = solver_status()
    defaults: dict[str, dict[str, Any]] = {}
    for solver_id, info in SOLVER_REGISTRY.items():
        defaults[solver_id] = {
            "id": solver_id,
            "label": info["label"],
            "summary": info["summary"],
            "availability": status[solver_id]["availability"],
            "defaults": MidlayerDesignConfig().normalized().to_dict(),
            "schema": _COMMON_SCHEMA + [
                {"key": f"{solver_id}_iterations", "label": "Iterations", "type": "number", "step": 1, "min": 8, "max": 200},
                {"key": f"{solver_id}_penalty", "label": "Penalty", "type": "number", "step": 0.1, "min": 1.0, "max": 6.0},
            ],
        }
    return defaults


def _to_section_spec(section: MidlayerSectionInput | SectionSpec) -> SectionSpec:
    if isinstance(section, SectionSpec):
        return section
    return SectionSpec(
        level=section.level,
        name=f"section_{section.level}",
        outer_meshes=[section.outer_mesh],
        inner_meshes=[section.inner_mesh],
    )


class MidlayerAdapter:
    def __init__(self, solver_id: str):
        if solver_id not in SOLVER_REGISTRY:
            raise KeyError(f"unknown solver '{solver_id}'")
        self.solver_id = solver_id
        self.label = SOLVER_REGISTRY[solver_id]["label"]
        status = solver_status()[solver_id]["availability"]
        self.availability = SolverAvailability(
            installed=bool(status["installed"]),
            package=str(status["package"]),
            mode=str(status["mode"]),
            notes=str(status["notes"]),
        )

    def run(
        self,
        sections: list[MidlayerSectionInput | SectionSpec],
        config: dict[str, Any] | MidlayerDesignConfig | None = None,
        *,
        output_dir: str | Path | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> MidlayerDesignResult:
        specs = [_to_section_spec(section) for section in sections]
        raw = build_midlayer_designs(
            specs,
            solver_id=self.solver_id,
            config=coerce_config(config),
            output_dir=output_dir,
            progress_callback=progress_callback,
        )
        artifacts = [
            MidlayerArtifact(
                level=int(item["level"]),
                name=str(item["name"]),
                mesh=item["mesh"],
                output_path=Path(item["output_path"]) if item.get("output_path") else None,
                metadata=dict(item["metadata"]),
            )
            for item in raw["sections"]
        ]
        return MidlayerDesignResult(
            solver_id=self.solver_id,
            label=self.label,
            mode=str(raw["mode"]),
            availability=self.availability,
            config=MidlayerDesignConfig(**coerce_config(config)).normalized(),
            artifacts=artifacts,
            notes=str(raw["status"]["detail"]),
        )


def get_adapter(solver_id: str) -> MidlayerAdapter:
    return MidlayerAdapter(solver_id)


def get_solver_registry() -> dict[str, MidlayerAdapter]:
    return {solver_id: MidlayerAdapter(solver_id) for solver_id in SOLVER_REGISTRY}


def export_solver_artifacts(result: MidlayerDesignResult, output_dir: str | Path) -> list[Path]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for artifact in result.artifacts:
        out_path = out_dir / f"{artifact.name}.stl"
        artifact.mesh.export(out_path)
        paths.append(out_path)
    return paths


def build_midlayer_designs(
    sections: list[SectionSpec],
    *,
    solver_id: str,
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if solver_id not in SOLVER_REGISTRY:
        raise ValueError(f"unknown solver '{solver_id}'")
    if not sections:
        raise ValueError("no matching outer/inner sections were provided")

    merged = coerce_config(config)
    status = solver_status()[solver_id]
    generated: list[dict[str, Any]] = []
    out_root = Path(output_dir).resolve() if output_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    for section_index, section in enumerate(sections, start=1):
        if progress_callback is not None:
            progress_callback({
                "kind": "section-start",
                "solver": solver_id,
                "sectionIndex": section_index,
                "sectionCount": len(sections),
                "level": section.level,
            })
        mesh, metadata = _design_section_mesh(
            section,
            solver_id=solver_id,
            config=merged,
            progress_callback=progress_callback,
            section_index=section_index,
            section_count=len(sections),
        )
        out_path = None
        if out_root is not None:
            out_path = out_root / f"mid_{section.level}_{solver_id}.stl"
            mesh.export(out_path)
        if progress_callback is not None:
            progress_callback({
                "kind": "section-done",
                "solver": solver_id,
                "sectionIndex": section_index,
                "sectionCount": len(sections),
                "level": section.level,
                "faces": int(len(mesh.faces)),
            })
        generated.append({
            "level": section.level,
            "name": f"mid_{section.level}_{solver_id}",
            "mesh": mesh,
            "output_path": str(out_path) if out_path is not None else None,
            "metadata": metadata,
        })

    return {
        "solver": solver_id,
        "label": SOLVER_REGISTRY[solver_id]["label"],
        "mode": status["mode"],
        "status": status,
        "config": merged,
        "sections": generated,
    }


def _merged_vertices(meshes: list[trimesh.Trimesh]) -> np.ndarray:
    verts = [np.asarray(mesh.vertices, dtype=float) for mesh in meshes if mesh is not None and len(mesh.vertices)]
    if not verts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(verts)


def _axis_profile(meshes: list[trimesh.Trimesh], axis: int, sample_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = _merged_vertices(meshes)
    if len(vertices) == 0:
        raise ValueError("section meshes are empty")
    radial_axes = [idx for idx in range(3) if idx != axis]
    axis_values = vertices[:, axis]
    axis_span = max(1e-6, float(axis_values.max() - axis_values.min()))
    band = axis_span / max(10.0, len(sample_z) / 1.8)
    global_center = vertices[:, radial_axes].mean(axis=0)
    global_r = float(np.quantile(np.linalg.norm(vertices[:, radial_axes] - global_center, axis=1), 0.92))

    centers = np.zeros((len(sample_z), 2), dtype=float)
    radii = np.zeros(len(sample_z), dtype=float)
    for idx, z in enumerate(sample_z):
        mask = np.abs(axis_values - z) <= band
        pts = vertices[mask][:, radial_axes]
        if len(pts) < 16:
            order = np.argsort(np.abs(axis_values - z))[: max(16, min(160, len(vertices)))]
            pts = vertices[order][:, radial_axes]
        center = np.median(pts, axis=0) if len(pts) else global_center
        radial = np.linalg.norm(pts - center, axis=1) if len(pts) else np.array([global_r])
        centers[idx] = center
        radii[idx] = max(1e-6, float(np.quantile(radial, 0.90)))

    centers = ndimage.gaussian_filter1d(centers, sigma=1.0, axis=0, mode="nearest")
    radii = ndimage.gaussian_filter1d(radii, sigma=1.0, mode="nearest")
    return centers, radii


def _infer_axis_and_profiles(section: SectionSpec, sample_count: int) -> dict[str, Any]:
    outer_vertices = _merged_vertices(section.outer_meshes)
    inner_vertices = _merged_vertices(section.inner_meshes)
    if len(outer_vertices) == 0 or len(inner_vertices) == 0:
        raise ValueError("section meshes are empty")

    outer_bounds = np.array([outer_vertices.min(axis=0), outer_vertices.max(axis=0)], dtype=float)
    extents = outer_bounds[1] - outer_bounds[0]
    axis = int(np.argmax(extents))
    radial_axes = [idx for idx in range(3) if idx != axis]
    zmin = float(outer_bounds[0, axis])
    zmax = float(outer_bounds[1, axis])
    sample_z = np.linspace(zmin, zmax, sample_count)
    outer_centers, outer_radii = _axis_profile(section.outer_meshes, axis, sample_z)
    inner_centers, inner_radii = _axis_profile(section.inner_meshes, axis, sample_z)
    return {
        "axis": axis,
        "radial_axes": radial_axes,
        "outer_bounds": outer_bounds,
        "zmin": zmin,
        "zmax": zmax,
        "span": max(1e-6, zmax - zmin),
        "sample_z": sample_z,
        "outer_centers": outer_centers,
        "inner_centers": inner_centers,
        "outer_radii": outer_radii,
        "inner_radii": inner_radii,
    }


def _interp_profile(profile: dict[str, Any], coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_z = profile["sample_z"]
    outer_centers = np.column_stack([
        np.interp(coords, sample_z, profile["outer_centers"][:, 0]),
        np.interp(coords, sample_z, profile["outer_centers"][:, 1]),
    ])
    inner_centers = np.column_stack([
        np.interp(coords, sample_z, profile["inner_centers"][:, 0]),
        np.interp(coords, sample_z, profile["inner_centers"][:, 1]),
    ])
    outer_r = np.interp(coords, sample_z, profile["outer_radii"])
    inner_r = np.interp(coords, sample_z, profile["inner_radii"])
    return outer_centers, inner_centers, np.column_stack([outer_r, inner_r])


def _grid_for_profile(profile: dict[str, Any], resolution: int) -> dict[str, Any]:
    outer_bounds = profile["outer_bounds"]
    extents = outer_bounds[1] - outer_bounds[0]
    max_extent = float(np.max(extents))
    pitch = max(max_extent / resolution, 1e-6)
    mins = outer_bounds[0] - (pitch * 1.1)
    maxs = outer_bounds[1] + (pitch * 1.1)
    dims = np.maximum(8, np.ceil((maxs - mins) / pitch).astype(int))
    grid_indices = np.stack(np.indices(tuple(dims)), axis=-1).reshape(-1, 3)
    points = mins[None, :] + (grid_indices.astype(float) + 0.5) * pitch
    return {"pitch": pitch, "mins": mins, "maxs": maxs, "dims": dims, "points": points}


def _largest_components(mask: np.ndarray, keep: int = 3) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count <= keep:
        return mask
    sizes = np.bincount(labels.ravel())
    keep_labels = np.argsort(sizes)[-keep:]
    return np.isin(labels, list(keep_labels))


def _constrain_mesh_to_profile(mesh: trimesh.Trimesh, profile: dict[str, Any], clearance: float) -> None:
    if not len(mesh.vertices):
        return
    vertices = np.asarray(mesh.vertices, dtype=float)
    axis = profile["axis"]
    radial_axes = profile["radial_axes"]
    coords = vertices[:, axis]
    outer_centers, _inner_centers, radii = _interp_profile(profile, coords)
    outer_r = np.maximum(radii[:, 0] - clearance, 1e-6)
    inner_r = np.maximum(radii[:, 1] + (clearance * 0.5), 0.0)
    rel = vertices[:, radial_axes] - outer_centers
    radial = np.linalg.norm(rel, axis=1)
    safe = np.maximum(radial, 1e-9)
    direction = rel / safe[:, None]
    lo = np.minimum(inner_r, outer_r)
    hi = np.maximum(inner_r, outer_r)
    clamped = np.clip(radial, lo, hi)
    vertices[:, radial_axes[0]] = outer_centers[:, 0] + (direction[:, 0] * clamped)
    vertices[:, radial_axes[1]] = outer_centers[:, 1] + (direction[:, 1] * clamped)
    mesh.vertices = vertices


def _build_solver_score(points: np.ndarray, profile: dict[str, Any], *, solver_id: str, config: dict[str, Any]) -> np.ndarray:
    axis = profile["axis"]
    radial_axes = profile["radial_axes"]
    coords = points[:, axis]
    outer_centers, _inner_centers, radii = _interp_profile(profile, coords)
    outer_r = radii[:, 0]
    inner_r = radii[:, 1]
    radial_points = points[:, radial_axes]
    radial = np.linalg.norm(radial_points - outer_centers, axis=1)
    span = np.maximum(outer_r - inner_r, 1e-6)

    shell_sigma = np.maximum(span * 0.12, profile["span"] / config["resolution"])
    inner_shell = np.exp(-(((radial - inner_r) / shell_sigma) ** 2))
    outer_shell = np.exp(-((((outer_r) - radial) / shell_sigma) ** 2))

    normalized_height = np.clip((coords - profile["zmin"]) / profile["span"], 0.0, 1.0)
    bridge_count = config["bridge_count"] + (1 if solver_id == "dl4to" else 0)
    twist_turns = (
        0.55 + (0.004 * config["dl4to_iterations"])
        if solver_id == "dl4to"
        else 0.22 + (0.003 * config["pymoto_iterations"])
    )
    penalty = config["dl4to_penalty"] if solver_id == "dl4to" else config["pymoto_penalty"]
    bridge_score = np.zeros(len(points), dtype=float)
    theta = np.arctan2(
        radial_points[:, 1] - outer_centers[:, 1],
        radial_points[:, 0] - outer_centers[:, 0],
    )
    for idx in range(bridge_count):
        phase = (2.0 * math.pi * idx) / bridge_count
        angle = phase + (twist_turns * 2.0 * math.pi * (normalized_height - 0.5))
        target_radius = inner_r + span * (0.28 + 0.44 * np.sin(np.pi * normalized_height) ** penalty)
        arc = np.abs(np.angle(np.exp(1j * (theta - angle))))
        distance = np.sqrt((radial - target_radius) ** 2 + (arc * np.maximum(target_radius, shell_sigma)) ** 2)
        bridge_sigma = np.maximum(span * 0.11, shell_sigma * 1.2)
        bridge_score = np.maximum(bridge_score, np.exp(-((distance / bridge_sigma) ** 2)))

    ring_score = np.zeros(len(points), dtype=float)
    for frac in np.linspace(0.16, 0.84, config["axial_slices"]):
        target_radius = inner_r + span * (0.46 if solver_id == "dl4to" else 0.58)
        z_weight = np.exp(-(((normalized_height - frac) / 0.085) ** 2))
        ring_score = np.maximum(
            ring_score,
            np.exp(-(((radial - target_radius) / np.maximum(span * 0.10, shell_sigma)) ** 2)) * z_weight,
        )

    axial_wave = 0.5 + 0.5 * np.sin(np.pi * normalized_height)
    return (0.42 * (inner_shell + outer_shell)) + (1.05 * bridge_score) + (0.18 * ring_score) + (0.22 * axial_wave)


def _design_context(section: SectionSpec, *, solver_id: str, config: dict[str, Any]) -> dict[str, Any]:
    profile = _infer_axis_and_profiles(section, int(config["resolution"]) + 4)
    grid = _grid_for_profile(profile, int(config["resolution"]))
    pitch = float(grid["pitch"])
    points = np.asarray(grid["points"], dtype=float)
    dims = tuple(int(v) for v in grid["dims"])
    axis = profile["axis"]
    radial_axes = profile["radial_axes"]

    coords = points[:, axis]
    outer_centers, _inner_centers, radii = _interp_profile(profile, coords)
    outer_r = radii[:, 0]
    inner_r = radii[:, 1]
    radial = np.linalg.norm(points[:, radial_axes] - outer_centers, axis=1)

    domain = (
        (coords >= profile["zmin"])
        & (coords <= profile["zmax"])
        & (radial <= np.maximum(outer_r - pitch * 0.35, 0.0))
        & (radial >= inner_r + pitch * 0.35)
    )
    if not np.any(domain):
        raise ValueError(f"section {section.name} has no valid design domain")

    shell_band = domain & (
        (np.abs(radial - inner_r) <= (pitch * float(config["shell_thickness"])))
        | (np.abs(outer_r - radial) <= (pitch * float(config["shell_thickness"])))
    )
    score = _build_solver_score(points, profile, solver_id=solver_id, config=config)
    return {
        "profile": profile,
        "grid": grid,
        "pitch": pitch,
        "points": points,
        "dims": dims,
        "domain": domain.reshape(dims),
        "domain_flat": domain,
        "shell_band": shell_band.reshape(dims),
        "shell_flat": shell_band,
        "score": score.reshape(dims),
        "score_flat": score,
    }


class _PyMOTOTargetObjective(_pymoto_native.Module if _pymoto_native is not None else object):
    def __init__(self, target: np.ndarray, weight: np.ndarray):
        self.target = np.asarray(target, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self._diff = np.zeros_like(self.target)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._diff = np.asarray(x, dtype=float) - self.target
        return np.array(np.sum(self.weight * self._diff * self._diff), dtype=float)

    def _sensitivity(self, df_dy: np.ndarray) -> np.ndarray:
        return 2.0 * self.weight * self._diff * float(df_dy)


class _PyMOTOVolumeConstraint(_pymoto_native.Module if _pymoto_native is not None else object):
    def __init__(self, domain_mask: np.ndarray, max_fraction: float):
        self.domain_mask = np.asarray(domain_mask, dtype=bool)
        self.max_fraction = float(max_fraction)
        self._count = max(1, int(np.count_nonzero(self.domain_mask)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        active = np.asarray(x, dtype=float)[self.domain_mask]
        return np.array(np.mean(active) - self.max_fraction, dtype=float)

    def _sensitivity(self, df_dy: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(self.domain_mask, dtype=float)
        grad[self.domain_mask] = float(df_dy) / self._count
        return grad


def _native_pymoto_density(context: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    import pymoto

    dims = tuple(int(v) for v in context["dims"])
    domain_mask = np.asarray(context["domain_flat"], dtype=bool)
    shell_mask = np.asarray(context["shell_flat"], dtype=bool)
    score = np.asarray(context["score_flat"], dtype=float)
    if np.ptp(score[domain_mask]) <= 1e-12:
        score_norm = np.where(domain_mask, 1.0, 0.0)
    else:
        smin = float(score[domain_mask].min())
        smax = float(score[domain_mask].max())
        score_norm = np.clip((score - smin) / max(1e-9, smax - smin), 0.0, 1.0)

    penalty = max(1.0, float(config["pymoto_penalty"]))
    target = np.where(domain_mask, score_norm ** penalty, 0.0)
    target[shell_mask] = 1.0

    weight = np.where(domain_mask, 1.0 + (2.0 * score_norm), 8.0)
    weight[shell_mask] += 4.0

    domain = pymoto.VoxelDomain(dims[0], dims[1], dims[2])
    filter_radius = max(1.5, float(config["shell_thickness"]) * 1.25)
    x0 = np.where(domain_mask, np.clip(float(config["volume_fraction"]) * (0.45 + target), 0.001, 1.0), 0.0)
    xmin = np.zeros_like(x0)
    xmax = np.where(domain_mask, 1.0, 1e-9)

    x = pymoto.Signal("rho", state=x0, min=xmin, max=xmax)
    with pymoto.Network() as net:
        xf = pymoto.DensityFilter(domain, radius=filter_radius)(x)
        obj = _PyMOTOTargetObjective(target, weight)(xf)
        vol = _PyMOTOVolumeConstraint(domain_mask, float(config["volume_fraction"]))(xf)

    pymoto.minimize_mma(
        x,
        [obj, vol],
        function=net,
        maxit=int(config["pymoto_iterations"]),
        tolx=1e-3,
        tolf=1e-4,
        xmin=xmin,
        xmax=xmax,
        move=0.2,
        verbosity=0,
    )
    net.response()
    density = np.asarray(xf.state, dtype=float).reshape(dims)
    density[~np.asarray(context["domain"], dtype=bool)] = 0.0
    metadata = {
        "filterRadius": filter_radius,
        "iterations": int(config["pymoto_iterations"]),
        "objective": float(obj.state),
        "constraint": float(vol.state),
        "densityMean": float(np.mean(density[np.asarray(context["domain"], dtype=bool)])),
    }
    return density, metadata


def _native_pymoto_density_with_progress(
    context: dict[str, Any],
    config: dict[str, Any],
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    section: SectionSpec,
    section_index: int,
    section_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import pymoto

    dims = tuple(int(v) for v in context["dims"])
    domain_mask = np.asarray(context["domain_flat"], dtype=bool)
    shell_mask = np.asarray(context["shell_flat"], dtype=bool)
    score = np.asarray(context["score_flat"], dtype=float)
    if np.ptp(score[domain_mask]) <= 1e-12:
        score_norm = np.where(domain_mask, 1.0, 0.0)
    else:
        smin = float(score[domain_mask].min())
        smax = float(score[domain_mask].max())
        score_norm = np.clip((score - smin) / max(1e-9, smax - smin), 0.0, 1.0)

    penalty = max(1.0, float(config["pymoto_penalty"]))
    target = np.where(domain_mask, score_norm ** penalty, 0.0)
    target[shell_mask] = 1.0

    weight = np.where(domain_mask, 1.0 + (2.0 * score_norm), 8.0)
    weight[shell_mask] += 4.0

    domain = pymoto.VoxelDomain(dims[0], dims[1], dims[2])
    filter_radius = max(1.5, float(config["shell_thickness"]) * 1.25)
    x0 = np.where(domain_mask, np.clip(float(config["volume_fraction"]) * (0.45 + target), 0.001, 1.0), 0.0)
    xmin = np.zeros_like(x0)
    xmax = np.where(domain_mask, 1.0, 1e-9)

    x = pymoto.Signal("rho", state=x0, min=xmin, max=xmax)
    with pymoto.Network() as net:
        xf = pymoto.DensityFilter(domain, radius=filter_radius)(x)
        obj = _PyMOTOTargetObjective(target, weight)(xf)
        vol = _PyMOTOVolumeConstraint(domain_mask, float(config["volume_fraction"]))(xf)

    optimizer = pymoto.MMA(
        x,
        [obj, vol],
        net,
        xmin=xmin,
        xmax=xmax,
        move=0.2,
        verbosity=0,
    )
    max_iterations = int(config["pymoto_iterations"])
    tolx = 1e-3
    tolf = 1e-4
    xval = optimizer.x
    gcur = 0.0
    performed = 0
    if progress_callback is not None:
        progress_callback({
            "kind": "native-setup",
            "solver": "pymoto",
            "sectionIndex": section_index,
            "sectionCount": section_count,
            "level": section.level,
            "maxIterations": max_iterations,
        })
    while optimizer.iter < max_iterations:
        xnew, g, dg = optimizer.step(x=xval)
        performed += 1

        gprev, gcur = gcur, g
        rel_df = np.linalg.norm(gcur - gprev) / max(1e-12, np.linalg.norm(gcur))
        rel_stepsize = np.linalg.norm((xval - xnew) / optimizer.dx) / max(1e-12, np.linalg.norm(xval / optimizer.dx))
        if progress_callback is not None:
            progress_callback({
                "kind": "iteration",
                "solver": "pymoto",
                "sectionIndex": section_index,
                "sectionCount": section_count,
                "level": section.level,
                "iteration": performed,
                "maxIterations": max_iterations,
                "objective": float(g[0] if np.ndim(g) else g),
                "constraint": float(g[1] if np.ndim(g) and np.size(g) > 1 else vol.state),
                "relativeFunctionChange": float(rel_df),
                "relativeStep": float(rel_stepsize),
            })

        if rel_df < tolf or rel_stepsize < tolx:
            break
        xval = xnew
        optimizer.iter += 1

    net.response()
    density = np.asarray(xf.state, dtype=float).reshape(dims)
    density[~np.asarray(context["domain"], dtype=bool)] = 0.0
    metadata = {
        "filterRadius": filter_radius,
        "iterations": performed,
        "objective": float(obj.state),
        "constraint": float(vol.state),
        "densityMean": float(np.mean(density[np.asarray(context["domain"], dtype=bool)])),
    }
    return density, metadata


def _scaffold_density(context: dict[str, Any], *, solver_id: str, config: dict[str, Any]) -> np.ndarray:
    domain = np.asarray(context["domain"], dtype=bool)
    shell_band = np.asarray(context["shell_band"], dtype=bool)
    score = np.asarray(context["score"], dtype=float)
    threshold = float(np.quantile(score[domain], max(0.0, 1.0 - float(config["volume_fraction"]))))
    occupied = domain & (score >= threshold)
    occupied = occupied | shell_band
    return occupied.astype(float)


def _mask_to_mesh(mask: np.ndarray, context: dict[str, Any], *, solver_id: str, config: dict[str, Any], section: SectionSpec) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    pitch = float(context["pitch"])
    grid = context["grid"]
    profile = context["profile"]
    dims = tuple(int(v) for v in context["dims"])
    structure = ndimage.generate_binary_structure(3, 1)
    cleaned = ndimage.binary_closing(mask.astype(bool), structure=structure, iterations=1)
    cleaned = ndimage.binary_opening(cleaned, structure=structure, iterations=1)
    cleaned = _largest_components(cleaned, keep=3)

    voxel = trimesh.voxel.VoxelGrid(cleaned, transform=scale_and_translate(scale=pitch, translate=np.asarray(grid["mins"], dtype=float)))
    mesh = voxel.as_boxes()
    if hasattr(mesh, "unique_faces"):
        try:
            mesh.update_faces(mesh.unique_faces())
        except Exception:
            pass
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    if int(config["smoothing_passes"]) > 0 and len(mesh.vertices) > 0:
        trimesh.smoothing.filter_humphrey(
            mesh,
            alpha=0.12 if solver_id == "dl4to" else 0.08,
            beta=0.48 if solver_id == "dl4to" else 0.40,
            iterations=int(config["smoothing_passes"]),
        )
        mesh.merge_vertices()
    outer_vertices = _merged_vertices(section.outer_meshes)
    if len(outer_vertices):
        outer_bounds = np.array([outer_vertices.min(axis=0), outer_vertices.max(axis=0)], dtype=float)
        outer_center = outer_bounds.mean(axis=0)
        mesh_center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(outer_center - mesh_center)

        axis_index = profile["axis"]
        radial_axes = [idx for idx in range(3) if idx != axis_index]
        mesh_bounds = mesh.bounds
        scale_xyz = np.ones(3, dtype=float)
        for ax in radial_axes:
            outer_span = float(outer_bounds[1, ax] - outer_bounds[0, ax])
            mesh_span = float(mesh_bounds[1, ax] - mesh_bounds[0, ax])
            target_span = max(1e-6, outer_span - (pitch * 0.6))
            if mesh_span > target_span:
                scale_xyz[ax] = target_span / mesh_span
        if np.any(np.abs(scale_xyz - 1.0) > 1e-6):
            mesh.apply_translation(-outer_center)
            transform = np.eye(4, dtype=float)
            transform[0, 0] = scale_xyz[0]
            transform[1, 1] = scale_xyz[1]
            transform[2, 2] = scale_xyz[2]
            mesh.apply_transform(transform)
            mesh.apply_translation(outer_center)
    _constrain_mesh_to_profile(mesh, profile, clearance=max(pitch * 0.35, 0.75))
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"section {section.name} produced an empty {solver_id} scaffold")
    metadata = {
        "axis": "xyz"[profile["axis"]],
        "pitch": pitch,
        "voxel_shape": [int(v) for v in dims],
        "estimatedVolumeFraction": float(np.mean(mask[np.asarray(context["domain"], dtype=bool)])),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
    }
    return mesh, metadata


def _design_section_mesh(
    section: SectionSpec,
    *,
    solver_id: str,
    config: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    section_index: int = 1,
    section_count: int = 1,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    context = _design_context(section, solver_id=solver_id, config=config)
    status = solver_status()[solver_id]
    if solver_id == "pymoto" and status["mode"] == "native":
        density, metadata = _native_pymoto_density_with_progress(
            context,
            config,
            progress_callback=progress_callback,
            section=section,
            section_index=section_index,
            section_count=section_count,
        )
        quantile = float(np.quantile(density[np.asarray(context["domain"], dtype=bool)], max(0.0, 1.0 - float(config["volume_fraction"]))))
        mask = (density >= quantile) & np.asarray(context["domain"], dtype=bool)
        mask |= np.asarray(context["shell_band"], dtype=bool)
        mesh, mesh_meta = _mask_to_mesh(mask, context, solver_id=solver_id, config=config, section=section)
        mesh_meta.update(metadata)
        mesh_meta["mode"] = "native"
        return mesh, mesh_meta

    density = _scaffold_density(context, solver_id=solver_id, config=config)
    mesh, metadata = _mask_to_mesh(density > 0.5, context, solver_id=solver_id, config=config, section=section)
    metadata["mode"] = "scaffold"
    return mesh, metadata
