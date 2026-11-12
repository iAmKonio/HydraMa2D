#!/usr/bin/env python3
"""Regenerates the Gmsh test fixtures of tests/test_gmsh_fixtures.cpp.

The files in this directory were written by the official Gmsh SDK
(pip install gmsh; version recorded in README.md of this directory), so the
reader is tested against files produced by Gmsh itself, not only against the
test writer of tests/GmshTestWriter.hpp.

    python3 generate_fixtures.py      # run from this directory

Meshes are deliberately coarse (files of a few kB).
"""
import gmsh

MESH_SIZE = 0.25


def start(name):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)       # Frontal-Delaunay: deterministic
    gmsh.option.setNumber("Mesh.RandomFactor", 1e-9)
    gmsh.model.add(name)


def write(name, version, binary, **options):
    gmsh.option.setNumber("Mesh.MshFileVersion", version)
    gmsh.option.setNumber("Mesh.Binary", 1 if binary else 0)
    for key, value in options.items():
        gmsh.option.setNumber("Mesh." + key, value)
    gmsh.write(name)


def rectangle(x0, y0, lx, ly, h, reverse=False):
    g = gmsh.model.geo
    p = [g.addPoint(x0, y0, 0, h), g.addPoint(x0 + lx, y0, 0, h),
         g.addPoint(x0 + lx, y0 + ly, 0, h), g.addPoint(x0, y0 + ly, 0, h)]
    lines = [g.addLine(p[0], p[1]), g.addLine(p[1], p[2]), g.addLine(p[2], p[3]), g.addLine(p[3], p[0])]
    loop = g.addCurveLoop([-l for l in reversed(lines)] if reverse else lines)
    return lines, loop


# 1. Channel [0,4] x [0,1] with a cylinder hole: curved inner boundary,
#    names with spaces / camelCase, a curve in two physical groups.
def channel_cylinder():
    start("channel_cylinder")
    g = gmsh.model.geo
    (bottom, right, top, left), outer = rectangle(0.0, 0.0, 4.0, 1.0, MESH_SIZE)
    c = g.addPoint(1.0, 0.5, 0, 0.08)
    q = [g.addPoint(1.0 + 0.2, 0.5, 0, 0.08), g.addPoint(1.0, 0.7, 0, 0.08),
         g.addPoint(1.0 - 0.2, 0.5, 0, 0.08), g.addPoint(1.0, 0.3, 0, 0.08)]
    arcs = [g.addCircleArc(q[i], c, q[(i + 1) % 4]) for i in range(4)]
    hole = g.addCurveLoop(arcs)
    surface = g.addPlaneSurface([outer, hole])
    g.synchronize()
    gmsh.model.addPhysicalGroup(1, [left], 1, "inlet")
    gmsh.model.addPhysicalGroup(1, [right], 2, "outlet")
    gmsh.model.addPhysicalGroup(1, [top, bottom], 3, "Symmetry Plane")
    gmsh.model.addPhysicalGroup(1, arcs, 4, "cylinderWall")
    gmsh.model.addPhysicalGroup(1, arcs, 5, "noSlip")      # same curves, same condition
    gmsh.model.addPhysicalGroup(2, [surface], 10, "fluid")
    gmsh.model.mesh.generate(2)
    for version in (2.2, 4.1):
        for binary in (False, True):
            write("channel_cylinder_%s_%s.msh" % (version, "binary" if binary else "ascii"), version, binary)
    gmsh.finalize()


# 2. Unit square variants.
def square(name, groups=True, reverse=False, unnamed=False):
    start(name)
    lines, loop = rectangle(0.0, 0.0, 1.0, 1.0, 0.4, reverse)
    surface = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    if groups:
        names = ["bottom_wall", "outlet", "top_wall", "inlet"]
        for i, line in enumerate(lines):
            gmsh.model.addPhysicalGroup(1, [line], 21 + i, "" if unnamed else names[i])
        gmsh.model.addPhysicalGroup(2, [surface], 1, "" if unnamed else "fluid")
    return surface


def squares():
    square("square")
    gmsh.model.mesh.generate(2)
    write("square_4.1_ascii.msh", 4.1, False)
    write("square_parametric_4.1_ascii.msh", 4.1, False, SaveParametric=1)
    gmsh.option.setNumber("Mesh.SaveParametric", 0)
    gmsh.model.mesh.setOrder(2)
    write("square_order2_4.1_ascii.msh", 4.1, False)
    gmsh.finalize()

    square("square_quads")
    gmsh.model.mesh.setRecombine(2, 1)
    gmsh.model.mesh.generate(2)
    write("square_quads_2.2_ascii.msh", 2.2, False)
    gmsh.finalize()

    # Curve loop reversed: the surface normal is -z, Gmsh writes clockwise triangles.
    square("square_clockwise", reverse=True)
    gmsh.model.mesh.generate(2)
    write("square_clockwise_2.2_ascii.msh", 2.2, False)
    gmsh.finalize()

    square("square_unnamed", unnamed=True)
    gmsh.model.mesh.generate(2)
    write("square_unnamed_4.1_binary.msh", 4.1, True)
    gmsh.finalize()

    # No physical groups at all: Gmsh saves every element (Mesh.SaveAll).
    square("square_nogroups", groups=False)
    gmsh.model.mesh.generate(2)
    write("square_nogroups_4.1_ascii.msh", 4.1, False, SaveAll=1)
    gmsh.finalize()


# 3. A 3D mesh (tetrahedra).
def cube():
    start("cube")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.6)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
    gmsh.model.mesh.generate(3)
    write("cube_tet_4.1_ascii.msh", 4.1, False, SaveAll=1)
    gmsh.finalize()


# 4. Hybrid mesh: quadrilateral inflation layer around a cylinder (Gmsh
#    BoundaryLayer field with Quads = 1), triangles elsewhere.
def inflation_layer():
    start("cylinder_inflation")
    g = gmsh.model.geo
    (bottom, right, top, left), outer = rectangle(0.0, 0.0, 4.0, 1.0, 0.2)
    c = g.addPoint(1.0, 0.5, 0, 0.1)
    q = [g.addPoint(1.0 + 0.2, 0.5, 0, 0.1), g.addPoint(1.0, 0.7, 0, 0.1),
         g.addPoint(1.0 - 0.2, 0.5, 0, 0.1), g.addPoint(1.0, 0.3, 0, 0.1)]
    arcs = [g.addCircleArc(q[i], c, q[(i + 1) % 4]) for i in range(4)]
    surface = g.addPlaneSurface([outer, g.addCurveLoop(arcs)])
    g.synchronize()
    bl = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(bl, "CurvesList", arcs)
    gmsh.model.mesh.field.setNumber(bl, "Size", 0.01)       # first cell height
    gmsh.model.mesh.field.setNumber(bl, "Ratio", 1.4)
    gmsh.model.mesh.field.setNumber(bl, "Thickness", 0.06)
    gmsh.model.mesh.field.setNumber(bl, "Quads", 1)
    gmsh.model.mesh.field.setAsBoundaryLayer(bl)
    gmsh.model.addPhysicalGroup(1, [left], 1, "inlet")
    gmsh.model.addPhysicalGroup(1, [right], 2, "outlet")
    gmsh.model.addPhysicalGroup(1, [top, bottom], 3, "symmetry")
    gmsh.model.addPhysicalGroup(1, arcs, 4, "cylinder_wall")
    gmsh.model.addPhysicalGroup(2, [surface], 10, "fluid")
    gmsh.model.mesh.generate(2)
    write("cylinder_inflation_4.1_binary.msh", 4.1, True)
    gmsh.finalize()


# 5. Structured quadrilateral channel [0,4] x [0,1] (transfinite, recombined).
def quad_channel():
    start("channel_quads")
    g = gmsh.model.geo
    (bottom, right, top, left), loop = rectangle(0.0, 0.0, 4.0, 1.0, 0.25)
    surface = g.addPlaneSurface([loop])
    for line, n in ((bottom, 17), (top, 17), (left, 6), (right, 6)):
        g.mesh.setTransfiniteCurve(line, n)
    g.mesh.setTransfiniteSurface(surface)
    g.mesh.setRecombine(2, surface)
    g.synchronize()
    gmsh.model.addPhysicalGroup(1, [left], 1, "inlet")
    gmsh.model.addPhysicalGroup(1, [right], 2, "outlet")
    gmsh.model.addPhysicalGroup(1, [top, bottom], 3, "wall")
    gmsh.model.addPhysicalGroup(2, [surface], 10, "fluid")
    gmsh.model.mesh.generate(2)
    write("channel_quads_2.2_ascii.msh", 2.2, False)
    gmsh.finalize()


if __name__ == "__main__":
    channel_cylinder()
    squares()
    cube()
    inflation_layer()
    quad_channel()
    gmsh.initialize()
    print("fixtures written with gmsh", gmsh.option.getString("General.Version"))
    gmsh.finalize()
