# Gmsh test fixtures

Written by the official Gmsh SDK, **version 4.15.2** (`pip install gmsh`), with `generate_fixtures.py` in this
directory. Used by `tests/test_gmsh_fixtures.cpp`, `tests/test_mesh_import_regression.cpp`, `tests/test_quad_solver.cpp`,
`tests/test_monitors.cpp`, `tests/test_cylinder_benchmark.cpp` and `tests/test_case_mesh_check.cpp`.

| File | Content | Expected |
| :--- | :--- | :--- |
| `channel_cylinder_{2.2,4.1}_{ascii,binary}.msh` | Channel [0,4]x[0,1] with a cylinder hole (r = 0.2). Groups: `inlet`, `outlet`, `Symmetry Plane`, `cylinderWall`, plus `noSlip` on the same curves as `cylinderWall`, and the surface group `fluid` | 246 triangles, 151 nodes, 56 sides, identical in all 4 variants |
| `square_4.1_ascii.msh` | Unit square, `bottom_wall` / `outlet` / `top_wall` / `inlet` | reference for the parametric file |
| `square_parametric_4.1_ascii.msh` | Same mesh, saved with parametric coordinates | same mesh |
| `square_clockwise_2.2_ascii.msh` | Surface with reversed curve loop: Gmsh writes every triangle clockwise | all 26 triangles reoriented |
| `square_unnamed_4.1_binary.msh` | Physical curve groups 21-24 without names | rejected; accepted with `*.bofl` entries |
| `square_nogroups_4.1_ascii.msh` | No physical groups (`Mesh.SaveAll`) | rejected: untagged boundary |
| `square_quads_2.2_ascii.msh` | Unit square, recombined (unstructured) quadrangles | 21 quadrilaterals, 30 nodes, 16 sides |
| `channel_quads_2.2_ascii.msh` | Channel [0,4]x[0,1], transfinite 16 x 5 quadrangles; groups `inlet`, `outlet`, `wall` | 80 quadrilaterals, 102 nodes, 42 sides |
| `cylinder_inflation_4.1_binary.msh` | Channel with a cylinder hole (r = 0.2, 16 wall segments) and a `BoundaryLayer` field with `Quads = 1`: 3 layers of 16 quadrangles around the cylinder, triangles elsewhere. Groups `inlet`, `outlet`, `symmetry`, `cylinder_wall` | 318 triangles, 48 quadrilaterals, 240 nodes, 66 sides; the 16 wall sides belong to quadrilaterals |
| `schaefer_turek_tri_L0.msh` | Schäfer–Turek cylinder channel (D = 1: [0, 22] x [0, 4.1], cylinder at (2, 2)), level 0 of `tools/cylinder_benchmark/generate_meshes.py`: triangles. Groups `parabolic_inlet`, `outlet`, `channel_wall`, `cylinder_wall` (16 sides) | 977 triangles, 534 nodes |
| `schaefer_turek_bl_L0.msh` | The same with a quadrilateral inflation layer (4 layers) on the cylinder | 951 triangles, 64 quadrilaterals, 585 nodes |
| `square_order2_4.1_ascii.msh` | Second-order elements | rejected: higher-order elements |
| `cube_tet_4.1_ascii.msh` | 3D tetrahedral mesh | rejected: 3D elements |

To regenerate: `cd tests/data/gmsh && python3 generate_fixtures.py`. The two `schaefer_turek_*_L0.msh` files come
from `python3 tools/cylinder_benchmark/generate_meshes.py OUT --levels 0` (renamed from `tri_L0.msh`, `bl_L0.msh`);
their benchmark values are pinned in `tests/test_cylinder_benchmark.cpp`. The meshing algorithm is fixed (Frontal-Delaunay)
for reproducibility. A different Gmsh version may still produce slightly different meshes, in which case the expected
counts in the tests must be updated.
