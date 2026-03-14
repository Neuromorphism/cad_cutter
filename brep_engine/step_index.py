"""Fast textual STEP scan helpers for preview-engine research."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_ENTITY_RE = re.compile(r"=\s*([A-Z0-9_]+)\s*\(")
_SCHEMA_RE = re.compile(r"FILE_SCHEMA\s*\(\(\s*'([^']+)'", re.IGNORECASE)
_TESSELLATED_PREFIXES = (
    "TESSELLATED_",
    "TRIANGULATED_",
    "COMPLEX_TRIANGULATED_",
)
_TESSELLATED_EXACT = {
    "FACETED_BREP",
    "FACETED_BREP_SHAPE_REPRESENTATION",
    "POLY_LOOP",
    "POLYLINE",
}


@dataclass
class StepScanResult:
    path: str
    schema: str | None
    entity_count: int
    top_entities: list[tuple[str, int]]
    has_tessellated_representation: bool
    tessellated_entities: dict[str, int]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _is_tessellated_entity(entity_name: str) -> bool:
    return (
        entity_name.startswith(_TESSELLATED_PREFIXES)
        or entity_name in _TESSELLATED_EXACT
    )


def scan_step_file(path: str | Path, top_n: int = 20) -> StepScanResult:
    target = Path(path).resolve()
    counts: Counter[str] = Counter()
    schema = None

    with target.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if schema is None:
                match = _SCHEMA_RE.search(line)
                if match:
                    schema = match.group(1)
            match = _ENTITY_RE.search(line)
            if match:
                counts[match.group(1)] += 1

    tessellated = {
        name: count for name, count in counts.items()
        if _is_tessellated_entity(name)
    }
    return StepScanResult(
        path=str(target),
        schema=schema,
        entity_count=sum(counts.values()),
        top_entities=counts.most_common(top_n),
        has_tessellated_representation=bool(tessellated),
        tessellated_entities=tessellated,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Scan a STEP file for preview-engine planning.")
    parser.add_argument("path", help="STEP file to scan")
    parser.add_argument("--top", type=int, default=20, help="How many top entity counts to print")
    args = parser.parse_args()
    print(scan_step_file(args.path, top_n=args.top).to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
