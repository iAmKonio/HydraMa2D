#!/usr/bin/env python3
"""Meshes of the Schafer-Turek "flow around a cylinder" benchmark (2D-1, 2D-2).

Schafer, M. & Turek, S. (1996), Benchmark computations of laminar flow around
a cylinder. Non-dimensional with the cylinder diameter D = 0.1 m as the unit
of length: channel [0, 22] x [0, 4.1], cylinder centre (2, 2), radius 0.5.

Two families with the same boundary and far-field sizes (docs/CYLINDER_BENCHMARK.md):
  tri  triangles only, graded from the cylinder;
  bl   the same, plus a quadrilateral inflation layer on the cylinder (Gmsh
       BoundaryLayer field with Quads = 1).

Physical groups: parabolic_inlet (left, flag 508), outlet (right, 504),
channel_wall (top and bottom, 500), cylinder_wall (500), fluid (surface).

Usage (needs the Gmsh Python module, `pip install gmsh`):
    generate_meshes.py OUT_DIR [--levels 1 2 3] [--families tri bl]
writes OUT_DIR/<family>_L<level>.msh (MSH 4.1 binary).
"""

import argparse
import math
import os
import sys

try:
    import gmsh
except ImportError:  # pragma: no cover
    sys.exit("generate_meshes.py needs the gmsh Python module: pip install gmsh")

LENGTH, HEIGHT = 22.0, 4.1
CX, CY, R = 2.0, 2.0, 0.5


def level_parameters(level):
    """Sizes of refinement level 1, 2, 3, ...: halved per level."""
    f = 2.0 ** (level - 1)
    n = int(32 * f)                       # segments on the cylinder
    return {
        "segments": n,
        "h_cylinder": 2.0 * math.pi * R / n,
        "h_wake": 0.16 / f,               # behind the cylinder (x < 10)
        "h_far": 0.4 / f,
        "bl_first": 0.01 / f,             # first layer height
        "bl_ratio": 1.2,
        "bl_thickness": 0.12,              # total inflation layer thickness, 0.12 D
    }


def build(family, level, path):
    p = level_parameters(level)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.add(f"{family}_L{level}")
    g = gmsh.model.geo
    corners = [g.addPoint(0, 0, 0), g.addPoint(LENGTH, 0, 0), g.addPoint(LENGTH, HEIGHT, 0), g.addPoint(0, HEIGHT, 0)]
    bottom, right, top, left = (g.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4))
    outer = g.addCurveLoop([bottom, right, top, left])
    c = g.addPoint(CX, CY, 0)
    q = [g.addPoint(CX + R, CY, 0), g.addPoint(CX, CY + R, 0), g.addPoint(CX - R, CY, 0), g.addPoint(CX, CY - R, 0)]
    arcs = [g.addCircleArc(q[i], c, q[(i + 1) % 4]) for i in range(4)]
    surface = g.addPlaneSurface([outer, g.addCurveLoop(arcs)])
    for a in arcs:
        g.mesh.setTransfiniteCurve(a, p["segments"] // 4 + 1)
    g.synchronize()

    f = gmsh.model.mesh.field
    dist = f.add("Distance")
    f.setNumbers(dist, "CurvesList", arcs)
    f.setNumber(dist, "Sampling", 200)
    grading = f.add("Threshold")
    f.setNumber(grading, "InField", dist)
    f.setNumber(grading, "SizeMin", p["h_cylinder"])
    f.setNumber(grading, "SizeMax", p["h_far"])
    f.setNumber(grading, "DistMin", 0.15)
    f.setNumber(grading, "DistMax", 3.0)
    wake = f.add("Box")
    f.setNumber(wake, "VIn", p["h_wake"])
    f.setNumber(wake, "VOut", p["h_far"])
    f.setNumber(wake, "XMin", CX)
    f.setNumber(wake, "XMax", 10.0)
    f.setNumber(wake, "YMin", CY - 1.0)
    f.setNumber(wake, "YMax", CY + 1.0)
    f.setNumber(wake, "Thickness", 1.0)
    size = f.add("Min")
    f.setNumbers(size, "FieldsList", [grading, wake])
    f.setAsBackgroundMesh(size)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.RandomSeed", 1)

    if family == "bl":
        bl = f.add("BoundaryLayer")
        f.setNumbers(bl, "CurvesList", arcs)
        f.setNumber(bl, "Size", p["bl_first"])
        f.setNumber(bl, "Ratio", p["bl_ratio"])
        f.setNumber(bl, "Thickness", p["bl_thickness"])
        f.setNumber(bl, "Quads", 1)
        f.setAsBoundaryLayer(bl)
    elif family != "tri":
        raise ValueError(family)

    gmsh.model.addPhysicalGroup(1, [left], 1, "parabolic_inlet")
    gmsh.model.addPhysicalGroup(1, [right], 2, "outlet")
    gmsh.model.addPhysicalGroup(1, [bottom, top], 3, "channel_wall")
    gmsh.model.addPhysicalGroup(1, arcs, 4, "cylinder_wall")
    gmsh.model.addPhysicalGroup(2, [surface], 10, "fluid")
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(path)
    nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--families", nargs="+", default=["tri", "bl"], choices=["tri", "bl"])
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for family in args.families:
        for level in args.levels:
            path = os.path.join(args.out_dir, f"{family}_L{level}.msh")
            nodes = build(family, level, path)
            print(f"{path}: {nodes} nodes")


if __name__ == "__main__":
    main()
