#!/usr/bin/env python3
"""Thermal face-coloring for WRL, STL, and 3MF meshes."""

import argparse
import math
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import xml.etree.ElementTree as ET


MATERIAL_DB = {
    "stainless_steel": {"diffusivity": 4.0e-6, "density": 8000.0, "cp": 500.0},
    "aluminum": {"diffusivity": 9.7e-5, "density": 2700.0, "cp": 900.0},
}


@dataclass
class Mesh:
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


class MeshThermalColorizer:
    PALETTES = {
        "red-blue": ((255, 0, 0), (0, 0, 255)),
        "fire-ice": ((255, 96, 0), (32, 96, 255)),
        "sunset": ((255, 72, 72), (61, 90, 254)),
        "viridis-ish": ((253, 231, 37), (68, 1, 84)),
        "grayscale": ((245, 245, 245), (20, 20, 20)),
    }

    def __init__(self, source_color=(255, 0, 0), sink_color=(0, 0, 255), mode="top-bottom",
                 source_temp=500.0, sink_temp=300.0, ambient_temp=300.0, material="stainless_steel",
                 dt=0.1, max_steps=4000, diffusion_rate=1.0, source_band=0.03, sink_band=0.03,
                 radial_inner=0.1, radial_outer=0.95, palette=None, reverse_palette=False):
        if palette:
            pal_source, pal_sink = self.palette_colors(palette)
            source_color, sink_color = pal_source, pal_sink
        if reverse_palette:
            source_color, sink_color = sink_color, source_color
        self.source_color = source_color
        self.sink_color = sink_color
        self.mode = mode
        self.source_temp = source_temp
        self.sink_temp = sink_temp
        self.ambient_temp = ambient_temp
        self.dt = dt
        self.max_steps = max_steps
        self.diffusion_rate = max(0.0, diffusion_rate)
        self.source_band = min(max(source_band, 0.0), 0.49)
        self.sink_band = min(max(sink_band, 0.0), 0.49)
        self.radial_inner = min(max(radial_inner, 0.0), 0.99)
        self.radial_outer = min(max(radial_outer, 0.01), 1.0)
        if self.radial_inner >= self.radial_outer:
            raise ValueError("radial_inner must be less than radial_outer")
        self.diffusivity = MATERIAL_DB.get(material, MATERIAL_DB["stainless_steel"])["diffusivity"]

    @staticmethod
    def hex_color(value: str) -> Tuple[int, int, int]:
        value = value.lstrip("#")
        if len(value) != 6:
            raise argparse.ArgumentTypeError(f"Invalid hex color: {value}")
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _interp(hot, cold, t):
        t = max(0.0, min(1.0, t))
        return tuple(int((1 - t) * cold[i] + t * hot[i]) for i in range(3))

    @classmethod
    def palette_colors(cls, name: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        if name not in cls.PALETTES:
            names = ", ".join(sorted(cls.PALETTES.keys()))
            raise ValueError(f"Unknown palette '{name}'. Expected one of: {names}")
        return cls.PALETTES[name]

    def load_mesh(self, filepath: str) -> Mesh:
        ext = Path(filepath).suffix.lower()
        if ext in {".wrl", ".vrml"}:
            return self._load_wrl(filepath)
        if ext == ".stl":
            return self._load_stl(filepath)
        if ext == ".3mf":
            return self._load_3mf(filepath)
        raise ValueError(f"Unsupported mesh format: {ext}")

    def _load_wrl(self, filepath: str) -> Mesh:
        text = Path(filepath).read_text()
        lower = text.lower()
        p0 = lower.find("point")
        a = text.find("[", p0)
        b = text.find("]", a)
        nums = [float(x) for x in text[a + 1:b].replace(",", " ").split()]
        vertices = [(nums[i], nums[i + 1], nums[i + 2]) for i in range(0, len(nums), 3)]

        p1 = lower.find("coordindex")
        c = text.find("[", p1)
        d = text.find("]", c)
        idxs = [int(x) for x in text[c + 1:d].replace(",", " ").split()]
        faces, poly = [], []
        for i in idxs:
            if i == -1:
                if len(poly) >= 3:
                    for k in range(1, len(poly) - 1):
                        faces.append((poly[0], poly[k], poly[k + 1]))
                poly = []
            else:
                poly.append(i)
        return Mesh(vertices, faces)

    def _load_stl(self, filepath: str) -> Mesh:
        with open(filepath, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(80)
            count_data = f.read(4)
        if len(count_data) == 4:
            tri_count = struct.unpack("<I", count_data)[0]
            if 84 + tri_count * 50 == size:
                return self._load_binary_stl(filepath)
        return self._load_ascii_stl(filepath)

    def _load_binary_stl(self, filepath: str) -> Mesh:
        vertices, faces, vmap = [], [], {}

        def get_idx(v):
            key = tuple(int(round(c * 1e6)) for c in v)
            if key in vmap:
                return vmap[key]
            idx = len(vertices)
            vertices.append(v)
            vmap[key] = idx
            return idx

        with open(filepath, "rb") as f:
            f.read(80)
            count = struct.unpack("<I", f.read(4))[0]
            for _ in range(count):
                f.read(12)
                tri = [struct.unpack("<fff", f.read(12)) for _ in range(3)]
                f.read(2)
                faces.append(tuple(get_idx(v) for v in tri))
        return Mesh(vertices, faces)

    def _load_ascii_stl(self, filepath: str) -> Mesh:
        vertices, faces, vmap, tri = [], [], {}, []

        def get_idx(v):
            key = tuple(int(round(c * 1e6)) for c in v)
            if key in vmap:
                return vmap[key]
            idx = len(vertices)
            vertices.append(v)
            vmap[key] = idx
            return idx

        for line in Path(filepath).read_text(errors="ignore").splitlines():
            s = line.strip().lower()
            if s.startswith("vertex"):
                p = line.split()
                tri.append(get_idx((float(p[1]), float(p[2]), float(p[3]))))
                if len(tri) == 3:
                    faces.append(tuple(tri))
                    tri = []
        return Mesh(vertices, faces)

    def _load_3mf(self, filepath: str) -> Mesh:
        with zipfile.ZipFile(filepath) as zf:
            model = next((n for n in zf.namelist() if n.startswith("3D/") and n.endswith(".model")), None)
            if not model:
                raise ValueError("3MF missing model")
            root = ET.fromstring(zf.read(model))
        ns = {"m": root.tag.split("}")[0].strip("{")}
        mesh = root.find(".//m:mesh", ns)
        if mesh is None:
            raise ValueError("3MF has no mesh")
        vertices = []
        for v in mesh.findall("m:vertices/m:vertex", ns):
            vertices.append((float(v.attrib["x"]), float(v.attrib["y"]), float(v.attrib["z"])))
        faces = []
        for t in mesh.findall("m:triangles/m:triangle", ns):
            faces.append((int(t.attrib["v1"]), int(t.attrib["v2"]), int(t.attrib["v3"])))
        return Mesh(vertices, faces)

    @staticmethod
    def face_centers(mesh: Mesh) -> List[Tuple[float, float, float]]:
        out = []
        for a, b, c in mesh.faces:
            va, vb, vc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
            out.append(((va[0] + vb[0] + vc[0]) / 3, (va[1] + vb[1] + vc[1]) / 3, (va[2] + vb[2] + vc[2]) / 3))
        return out

    def source_sink_sets(self, mesh: Mesh, centers: List[Tuple[float, float, float]]) -> Tuple[Set[int], Set[int]]:
        src, sink = set(), set()
        if self.mode == "top-bottom":
            zs = [c[2] for c in centers]
            zmin, zmax = min(zs), max(zs)
            span = max(1e-12, zmax - zmin)
            zsrc, zsink = zmax - self.source_band * span, zmin + self.sink_band * span
            for i, c in enumerate(centers):
                if c[2] >= zsrc:
                    src.add(i)
                if c[2] <= zsink:
                    sink.add(i)
        elif self.mode == "side-side":
            xs = [c[0] for c in centers]
            xmin, xmax = min(xs), max(xs)
            span = max(1e-12, xmax - xmin)
            xsrc, xsink = xmax - self.source_band * span, xmin + self.sink_band * span
            for i, c in enumerate(centers):
                if c[0] >= xsrc:
                    src.add(i)
                if c[0] <= xsink:
                    sink.add(i)
        elif self.mode == "front-back":
            ys = [c[1] for c in centers]
            ymin, ymax = min(ys), max(ys)
            span = max(1e-12, ymax - ymin)
            ysrc, ysink = ymax - self.source_band * span, ymin + self.sink_band * span
            for i, c in enumerate(centers):
                if c[1] >= ysrc:
                    src.add(i)
                if c[1] <= ysink:
                    sink.add(i)
        elif self.mode == "radial":
            cx = sum(c[0] for c in centers) / len(centers)
            cy = sum(c[1] for c in centers) / len(centers)
            vertex_rs = [math.sqrt((v[0] - cx) ** 2 + (v[1] - cy) ** 2) for v in mesh.vertices]
            R = max(1e-12, max(vertex_rs))
            for i, (a, b, c) in enumerate(mesh.faces):
                rmin = min(vertex_rs[a], vertex_rs[b], vertex_rs[c])
                rmax = max(vertex_rs[a], vertex_rs[b], vertex_rs[c])
                if rmin <= self.radial_inner * R:
                    sink.add(i)
                if rmax >= self.radial_outer * R:
                    src.add(i)
        else:
            raise ValueError("mode must be top-bottom, side-side, front-back, or radial")
        if not src or not sink:
            raise ValueError("Could not locate source/sink faces")
        return src, sink

    @staticmethod
    def adjacency(faces: Sequence[Tuple[int, int, int]]) -> List[Set[int]]:
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for fi, (a, b, c) in enumerate(faces):
            for u, v in ((a, b), (b, c), (c, a)):
                e = tuple(sorted((u, v)))
                edge_to_faces.setdefault(e, []).append(fi)
        adj = [set() for _ in faces]
        for lst in edge_to_faces.values():
            for i in lst:
                for j in lst:
                    if i != j:
                        adj[i].add(j)
        return adj

    def simulate(self, mesh: Mesh) -> List[float]:
        centers = self.face_centers(mesh)
        source, sink = self.source_sink_sets(mesh, centers)
        adj = self.adjacency(mesh.faces)

        T = [self.ambient_temp for _ in mesh.faces]
        for i in source:
            T[i] = self.source_temp
        for i in sink:
            T[i] = self.sink_temp

        mid = (self.source_temp + self.sink_temp) / 2
        scale = self.diffusivity * self.dt * 1e6 * self.diffusion_rate
        for step in range(self.max_steps):
            nxt = T[:]
            for i in range(len(T)):
                if i in source or i in sink or not adj[i]:
                    continue
                nbr_mean = sum(T[j] for j in adj[i]) / len(adj[i])
                nxt[i] = T[i] + scale * (nbr_mean - T[i])
            for i in source:
                nxt[i] = self.source_temp
            for i in sink:
                nxt[i] = self.sink_temp
            T = nxt

            frac = sum(1 for t in T if t <= mid) / len(T)
            if step > 30 and abs(frac - 0.5) < 0.03:
                break
        return T

    def face_colors(self, temperatures: Sequence[float]) -> List[Tuple[int, int, int]]:
        tmin, tmax = min(temperatures), max(temperatures)
        span = max(1e-12, tmax - tmin)
        return [self._interp(self.source_color, self.sink_color, (t - tmin) / span) for t in temperatures]

    @staticmethod
    def write_ply(mesh: Mesh, colors: Sequence[Tuple[int, int, int]], out_path: str) -> None:
        with open(out_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(mesh.vertices)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {len(mesh.faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for x, y, z in mesh.vertices:
                f.write(f"{x} {y} {z}\n")
            for (a, b, c), (r, g, bcol) in zip(mesh.faces, colors):
                f.write(f"3 {a} {b} {c} {r} {g} {bcol}\n")

    @staticmethod
    def write_svg_render(mesh: Mesh, colors: Sequence[Tuple[int, int, int]], out_svg: str, title: str) -> None:
        yaw, pitch = math.radians(35), math.radians(25)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)

        rotated = []
        for v in mesh.vertices:
            x1 = v[0] * cy - v[1] * sy
            y1 = v[0] * sy + v[1] * cy
            z1 = v[2]
            y2 = y1 * cp - z1 * sp
            z2 = y1 * sp + z1 * cp
            rotated.append((x1, y2, z2))

        xs, ys = [v[0] for v in rotated], [v[1] for v in rotated]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        pad = 20
        w, h = 800, 600
        sx = (w - 2 * pad) / max(1e-12, maxx - minx)
        sy2 = (h - 2 * pad) / max(1e-12, maxy - miny)
        s = min(sx, sy2)

        tris = []
        for i, (a, b, c) in enumerate(mesh.faces):
            p = [rotated[a], rotated[b], rotated[c]]
            depth = sum(v[2] for v in p) / 3
            pts = [((v[0] - minx) * s + pad, h - ((v[1] - miny) * s + pad)) for v in p]
            tris.append((depth, pts, colors[i]))
        tris.sort(key=lambda t: t[0])

        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
                 f'<rect width="100%" height="100%" fill="white"/>',
                 f'<text x="20" y="30" font-size="20" font-family="Arial">{title}</text>']
        for _, pts, c in tris:
            pstr = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            lines.append(f'<polygon points="{pstr}" fill="rgb({c[0]},{c[1]},{c[2]})" stroke="black" stroke-opacity="0.15" stroke-width="0.4"/>')
        lines.append("</svg>")
        Path(out_svg).parent.mkdir(parents=True, exist_ok=True)
        Path(out_svg).write_text("\n".join(lines))

    def process(self, input_path: str, output_ply: str, render_svg: str = None) -> None:
        mesh = self.load_mesh(input_path)
        temps = self.simulate(mesh)
        colors = self.face_colors(temps)
        self.write_ply(mesh, colors, output_ply)
        if render_svg:
            self.write_svg_render(mesh, colors, render_svg, f"{self.mode} thermal coloring")


def write_ascii_stl(mesh: Mesh, path: str) -> None:
    with open(path, "w") as f:
        f.write("solid mesh\n")
        for a, b, c in mesh.faces:
            v0, v1, v2 = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
            n = cross(v_sub(v1, v0), v_sub(v2, v0))
            nlen = max(1e-12, norm(n))
            n = v_scale(n, 1 / nlen)
            f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
            f.write("    outer loop\n")
            for v in (v0, v1, v2):
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid mesh\n")


def create_cube_mesh(size=1.0) -> Mesh:
    s = size / 2
    v = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
         (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]
    f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
         (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
         (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return Mesh(v, f)


def create_cylinder_mesh(radius=1.0, height=2.0, segments=48) -> Mesh:
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    z0, z1 = -height / 2, height / 2
    for i in range(segments):
        th = 2 * math.pi * i / segments
        x, y = radius * math.cos(th), radius * math.sin(th)
        verts.append((x, y, z0))
        verts.append((x, y, z1))
    bctr = len(verts)
    verts.append((0, 0, z0))
    tctr = len(verts)
    verts.append((0, 0, z1))

    for i in range(segments):
        j = (i + 1) % segments
        b0, t0, b1, t1 = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
        faces.extend([(b0, b1, t1), (b0, t1, t0), (bctr, b1, b0), (tctr, t0, t1)])
    return Mesh(verts, faces)


def generate_test_renders(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cube_stl = out / "cube.stl"
    cyl_stl = out / "cylinder.stl"
    write_ascii_stl(create_cube_mesh(), str(cube_stl))
    write_ascii_stl(create_cylinder_mesh(), str(cyl_stl))

    top = MeshThermalColorizer(mode="top-bottom")
    top.process(str(cube_stl), str(out / "cube_top_bottom_colored.ply"), str(out / "cube_top_bottom.svg"))

    radial = MeshThermalColorizer(mode="radial")
    radial.process(str(cyl_stl), str(out / "cylinder_radial_colored.ply"), str(out / "cylinder_radial.svg"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Thermal simulation face coloring for WRL/STL/3MF meshes")
    parser.add_argument("input", nargs="?", help="Input mesh (.wrl/.stl/.3mf)")
    parser.add_argument("-o", "--output", default="colored_output.ply", help="Output colored PLY")
    parser.add_argument("--render", default=None, help="Optional SVG render output path")
    parser.add_argument("--mode", choices=["top-bottom", "side-side", "front-back", "radial"], default="top-bottom")
    parser.add_argument("--palette", choices=sorted(MeshThermalColorizer.PALETTES.keys()), default=None,
                        help="Optional source/sink palette name")
    parser.add_argument("--reverse-palette", action="store_true", help="Swap source/sink palette colors")
    parser.add_argument("--source-color", type=MeshThermalColorizer.hex_color, default=(255, 0, 0))
    parser.add_argument("--sink-color", type=MeshThermalColorizer.hex_color, default=(0, 0, 255))
    parser.add_argument("--source-temp", type=float, default=500.0)
    parser.add_argument("--sink-temp", type=float, default=300.0)
    parser.add_argument("--ambient-temp", type=float, default=300.0)
    parser.add_argument("--material", choices=sorted(MATERIAL_DB.keys()), default="stainless_steel")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--diffusion-rate", type=float, default=1.0,
                        help="Multiplier on diffusion update (1.0 keeps legacy behavior)")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--source-band", type=float, default=0.03,
                        help="Fractional band from hot side used as source faces")
    parser.add_argument("--sink-band", type=float, default=0.03,
                        help="Fractional band from cold side used as sink faces")
    parser.add_argument("--radial-inner", type=float, default=0.1,
                        help="Inner radius fraction for radial sink selection")
    parser.add_argument("--radial-outer", type=float, default=0.95,
                        help="Outer radius fraction for radial source selection")
    parser.add_argument("--generate-test-renders", action="store_true")
    parser.add_argument("--test-output-dir", default="test_renders")
    args = parser.parse_args()

    if args.generate_test_renders:
        generate_test_renders(args.test_output_dir)
        print(f"✓ Test renders generated in {args.test_output_dir}")
        return

    if not args.input:
        parser.error("input is required unless --generate-test-renders is used")

    app = MeshThermalColorizer(
        source_color=args.source_color,
        sink_color=args.sink_color,
        mode=args.mode,
        source_temp=args.source_temp,
        sink_temp=args.sink_temp,
        ambient_temp=args.ambient_temp,
        material=args.material,
        dt=args.dt,
        diffusion_rate=args.diffusion_rate,
        max_steps=args.max_steps,
        source_band=args.source_band,
        sink_band=args.sink_band,
        radial_inner=args.radial_inner,
        radial_outer=args.radial_outer,
        palette=args.palette,
        reverse_palette=args.reverse_palette,
    )
    app.process(args.input, args.output, args.render)
    print(f"✓ Colored mesh written to {args.output}")
    if args.render:
        print(f"✓ Render written to {args.render}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
