"""Benchmark the current STEP preview path for research work."""

from __future__ import annotations

import json
import time
from pathlib import Path

import web_ui
from .step_index import scan_step_file


def benchmark_preview(path: str | Path) -> dict[str, object]:
    target = Path(path).resolve()
    preview = web_ui._find_preview_surrogate(target)
    load_path = preview or target

    load_started = time.perf_counter()
    wp, _name = web_ui._load_cached_part(str(load_path), require_solid=False)
    load_elapsed = time.perf_counter() - load_started

    part = web_ui.PartState(
        file_path=str(target),
        name=target.stem,
        source_ext=target.suffix.lower(),
        shape=wp.val().wrapped,
        mesh_source_path=str(load_path),
        preview_path=str(preview) if preview else None,
        preview_ext=preview.suffix.lower() if preview else None,
        preview_only=preview is not None,
    )
    web_ui.state.parts = [part]

    first_started = time.perf_counter()
    first_scene = web_ui._build_scene()
    first_scene_elapsed = time.perf_counter() - first_started

    second_started = time.perf_counter()
    second_scene = web_ui._build_scene()
    second_scene_elapsed = time.perf_counter() - second_started

    return {
        "path": str(target),
        "preview_path": str(preview) if preview else None,
        "load_seconds": round(load_elapsed, 3),
        "first_scene_seconds": round(first_scene_elapsed, 3),
        "second_scene_seconds": round(second_scene_elapsed, 3),
        "vertex_count": len(second_scene["parts"][0]["mesh"]["vertices"]) // 3 if second_scene["parts"] else 0,
        "combined_entries": len(first_scene["combined"]),
        "step_scan": scan_step_file(target).top_entities,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark STEP preview timings.")
    parser.add_argument("path", help="STEP file to benchmark")
    args = parser.parse_args()
    print(json.dumps(benchmark_preview(args.path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
