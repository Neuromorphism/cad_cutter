"""Optional topology-optimization adapters for generated midlayer meshes."""

from .adapters import (
    DEFAULT_CONFIG,
    SOLVER_REGISTRY,
    MidlayerAdapter,
    MidlayerArtifact,
    MidlayerDesignConfig,
    MidlayerDesignResult,
    MidlayerSectionInput,
    SectionSpec,
    SolverAvailability,
    build_midlayer_designs,
    coerce_config,
    export_solver_artifacts,
    get_adapter,
    get_solver_defaults,
    get_solver_registry,
    solver_status,
)

__all__ = [
    "DEFAULT_CONFIG",
    "SOLVER_REGISTRY",
    "MidlayerAdapter",
    "MidlayerArtifact",
    "MidlayerDesignConfig",
    "MidlayerDesignResult",
    "MidlayerSectionInput",
    "SectionSpec",
    "SolverAvailability",
    "build_midlayer_designs",
    "coerce_config",
    "export_solver_artifacts",
    "get_adapter",
    "get_solver_defaults",
    "get_solver_registry",
    "solver_status",
]
