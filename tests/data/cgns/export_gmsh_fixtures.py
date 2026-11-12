#!/usr/bin/env python3
"""Exports tests/data/gmsh meshes to CGNS with the Gmsh SDK (pip install gmsh).

Gmsh writes every boundary as a FamilySpecified BC_t whose family is named
after the geometric entity (L_1, L_2, ...; no FamilyBC), on PointRange /
CellCenter ranges of BAR_2 elements, plus a "BC" for the surface (S_1) on the
triangles. So these files carry no usable boundary names: they test a real
producer's layout and the *.bofl path (BC numbers).

    python3 export_gmsh_fixtures.py      # run from this directory
"""
import os
import gmsh

SRC = os.path.join("..", "gmsh")
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
for msh, cgns in [("square_4.1_ascii.msh", "square_gmsh.cgns"),
                  ("channel_cylinder_4.1_binary.msh", "channel_cylinder_gmsh.cgns")]:
    gmsh.clear()
    gmsh.open(os.path.join(SRC, msh))
    gmsh.write(cgns)
print("CGNS files written with gmsh", gmsh.option.getString("General.Version"))
gmsh.finalize()
