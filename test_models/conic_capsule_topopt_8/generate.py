#!/usr/bin/env python3
"""Generate a conic eight-section test model set for the CAD cutter pipeline."""

from __future__ import annotations

from pathlib import Path
import math

import cadquery as cq


OUT_DIR = Path(__file__).resolve().parent
SECTION_COUNT = 8
SECTION_HEIGHT = 500.0
TOTAL_HEIGHT = SECTION_COUNT * SECTION_HEIGHT
OUTER_DIAMETERS = [500.0, 470.0, 440.0, 405.0, 370.0, 335.0, 300.0, 260.0, 220.0]
INNER_DIAMETERS = [200.0, 188.0, 176.0, 164.0, 152.0, 140.0]

OUTER_WALL = 18.0
OUTER_CLEARANCE = 28.0
INNER_CLEARANCE = 18.0
CAPSULE_HEIGHT = 420.0


def capsule(radius: float, height: float) -> cq.Workplane:
    """Create a pill-shaped solid aligned to Z and centered on the origin."""
    cyl_height = max(height - 2.0 * radius, 1.0)
    body = cq.Workplane("XY").cylinder(cyl_height, radius)
    top = cq.Workplane("XY").transformed(offset=(0, 0, cyl_height / 2.0)).sphere(radius)
    bottom = cq.Workplane("XY").transformed(offset=(0, 0, -cyl_height / 2.0)).sphere(radius)
    return body.union(top).union(bottom)


def outer_shell(bottom_diameter: float, top_diameter: float, height: float) -> cq.Workplane:
    """Create a hollow frustum centered on the origin."""
    outer = cq.Workplane("XY").add(
        cq.Solid.makeCone(
            bottom_diameter / 2.0,
            top_diameter / 2.0,
            height,
            pnt=cq.Vector(0, 0, -height / 2.0),
            dir=cq.Vector(0, 0, 1),
        )
    )
    inner = cq.Workplane("XY").add(
        cq.Solid.makeCone(
            bottom_diameter / 2.0 - OUTER_WALL,
            top_diameter / 2.0 - OUTER_WALL,
            height,
            pnt=cq.Vector(0, 0, -height / 2.0),
            dir=cq.Vector(0, 0, 1),
        )
    )
    return outer.cut(inner)


def cylinder_between_points(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
) -> cq.Workplane:
    """Create a cylinder aligned between two arbitrary points."""
    start_vec = cq.Vector(*start)
    end_vec = cq.Vector(*end)
    delta = end_vec.sub(start_vec)
    return cq.Workplane("XY").add(
        cq.Solid.makeCylinder(radius, delta.Length, pnt=start_vec, dir=delta)
    )


def suspension_strut(
    start: tuple[float, float, float],
    mid: tuple[float, float, float],
    end: tuple[float, float, float],
    r0: float,
    r1: float,
    r2: float,
) -> cq.Workplane:
    """Create an organic support strut from blended cylinders and nodes."""
    leg0 = cylinder_between_points(start, mid, (r0 + r1) / 2.0)
    leg1 = cylinder_between_points(mid, end, (r1 + r2) / 2.0)
    node0 = cq.Workplane("XY").transformed(offset=start).sphere(r0)
    node1 = cq.Workplane("XY").transformed(offset=mid).sphere(r1)
    node2 = cq.Workplane("XY").transformed(offset=end).sphere(r2)
    return leg0.union(leg1).union(node0).union(node1).union(node2)


def suspension_section(
    outer_bottom_d: float,
    outer_top_d: float,
    inner_d: float,
    height: float,
    phase_deg: float,
) -> cq.Workplane:
    """Create a topology-optimization-inspired suspension between inner and outer."""
    outer_mid_r = min(outer_bottom_d, outer_top_d) / 2.0 - OUTER_WALL - OUTER_CLEARANCE
    inner_r = inner_d / 2.0 + INNER_CLEARANCE
    z0 = -height / 2.0 + 55.0
    z1 = 0.0
    z2 = height / 2.0 - 55.0

    result = None
    branch_count = 4
    for idx in range(branch_count):
        angle = math.radians(phase_deg + idx * 360.0 / branch_count)
        twist = angle + math.radians(24.0 if idx % 2 == 0 else -18.0)
        bend = angle + math.radians(42.0 if idx % 2 == 0 else -36.0)
        start = (inner_r * math.cos(angle), inner_r * math.sin(angle), z0)
        mid = (
            (inner_r + 0.45 * (outer_mid_r - inner_r)) * math.cos(twist),
            (inner_r + 0.45 * (outer_mid_r - inner_r)) * math.sin(twist),
            z1,
        )
        end = (outer_mid_r * math.cos(bend), outer_mid_r * math.sin(bend), z2)
        strut = suspension_strut(start, mid, end, 11.0, 16.0, 13.0)
        result = strut if result is None else result.union(strut)

    neck = (
        cq.Workplane("XY")
        .workplane(offset=-height / 2.0 + 120.0)
        .circle(inner_r * 0.72)
        .workplane(offset=height - 240.0)
        .circle(inner_r * 0.64)
        .loft(combine=True)
    )
    neck = neck.cut(
        cq.Workplane("XY")
        .workplane(offset=-height / 2.0 + 110.0)
        .circle(inner_r * 0.44)
        .workplane(offset=height - 220.0)
        .circle(inner_r * 0.36)
        .loft(combine=True)
    )
    result = result.union(neck)

    for ring_z, scale in ((-120.0, 1.0), (110.0, 0.88)):
        ring_outer = max(inner_r + 28.0, outer_mid_r * scale)
        ring_inner = max(inner_r + 10.0, ring_outer - 18.0)
        ring = (
            cq.Workplane("XY")
            .workplane(offset=ring_z)
            .circle(ring_outer)
            .circle(ring_inner)
            .extrude(18.0, both=True)
        )
        for slot_idx in range(6):
            slot_angle = math.radians(phase_deg + slot_idx * 60.0)
            cutter = (
                cq.Workplane("XY")
                .workplane(offset=ring_z)
                .center((ring_outer - 8.0) * math.cos(slot_angle), (ring_outer - 8.0) * math.sin(slot_angle))
                .rect(28.0, 90.0)
                .extrude(28.0, both=True)
            )
            ring = ring.cut(cutter.rotate((0, 0, ring_z), (0, 0, ring_z + 1.0), math.degrees(slot_angle)))
        result = result.union(ring)

    return result


def export_step(model: cq.Workplane, path: Path) -> None:
    cq.exporters.export(model, str(path))


def main() -> None:
    print(f"Generating test model into {OUT_DIR}")
    for idx in range(1, SECTION_COUNT + 1):
        outer = outer_shell(
            OUTER_DIAMETERS[idx - 1],
            OUTER_DIAMETERS[idx],
            SECTION_HEIGHT,
        )
        export_step(outer, OUT_DIR / f"outer_{idx}.step")
        print(
            f"  outer_{idx}.step: {OUTER_DIAMETERS[idx - 1]:.1f} mm -> "
            f"{OUTER_DIAMETERS[idx]:.1f} mm, h={SECTION_HEIGHT:.1f} mm"
        )

        if idx <= len(INNER_DIAMETERS):
            inner_d = INNER_DIAMETERS[idx - 1]
            inner = capsule(inner_d / 2.0, CAPSULE_HEIGHT)
            mid = suspension_section(
                OUTER_DIAMETERS[idx - 1],
                OUTER_DIAMETERS[idx],
                inner_d,
                SECTION_HEIGHT,
                phase_deg=idx * 11.0,
            )
            export_step(inner, OUT_DIR / f"inner_{idx}.step")
            export_step(mid, OUT_DIR / f"mid_{idx}.step")
            print(
                f"  inner_{idx}.step: d={inner_d:.1f} mm, h={CAPSULE_HEIGHT:.1f} mm"
            )
            print(f"  mid_{idx}.step: organic suspension")

    print(f"Total height represented by sections: {TOTAL_HEIGHT:.1f} mm")


if __name__ == "__main__":
    main()
