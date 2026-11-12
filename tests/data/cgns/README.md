# CGNS test fixtures

Used by `tests/test_cgns_reader.cpp` (skipped in builds without the CGNS library).

## Synthetic fixtures

Written by `generate_fixtures.cpp` with **CGNS 3.4.0 / HDF5 1.10**. CGNS 3.x refuses files written by CGNS 4.x,
and 4.x reads 3.x files, so these fixtures are readable by every supported version.

```bash
cmake --build build --target hydrama2d_cgns_fixtures          # a build configured with CGNS 3.4
cd tests/data/cgns && ../../../build/tests/hydrama2d_cgns_fixtures
```

The triangle channel meshes are `tests/data/su2/channel_regular.su2` (64 triangles, 45 vertices, 24 boundary edges). The
tests check that they import to exactly that mesh.

| File | Layout | Boundary conditions | Expected |
| :--- | :--- | :--- | :--- |
| `channel_sections.cgns` | HDF5; `TRI_3` section + one `BAR_2` section per boundary | `ElementRange` (stored as `PointRange` at `EdgeCenter` by CGNS 3.4); names `inlet` / `outlet` / `upper_wall` / `lower_wall`; BCTypes `BCInflowSubsonic`, `BCOutflow`, `BCWallViscous`, `BCWall` | valid; conditions from name + declared type |
| `channel_sections_adf.cgns` | the same in ADF | same | valid |
| `channel_mixed_vertex_bcs.cgns` | one `MIXED` section (bars, then triangles) | `PointList` at `Vertex`; names without keywords (`left`, `right`, `top`, `bottom`); BCTypes `BCInflow`, `BCOutflow`, `BCWall`, `BCSymmetryPlane` | valid; conditions from the declared types; bottom = symmetry |
| `channel_families_clockwise.cgns` | clockwise triangles; `PhysicalDimension` 3 with `CoordinateZ` | `FamilySpecified` BCs `BC_1`..`BC_4` in families `Inflow` / `Outflow` / `Walls` with `FamilyBC` types | valid; 64 triangles reoriented; names from the families |
| `channel_quads.cgns` | ADF; `QUAD_4` section (the 8 x 4 cells of the channel without diagonals) + `BAR_2` sections | `ElementRange`, as `channel_sections` | valid; 32 quadrilaterals, 45 vertices, 24 boundary edges |
| `unsupported_3d.cgns` | `CellDimension` 3, a `TETRA_4` section | - | "3D (CellDimension 3)" |
| `unsupported_tri6.cgns` | an extra `TRI_6` section | - | "higher-order elements" |
| `unsupported_ngon.cgns` | an extra `NGON_n` section | - | "polygonal / polyhedral elements" |
| `unsupported_mixed_hexa.cgns` | a `HEXA_8` inside the `MIXED` section | - | "1 HEXA_8 element: 3D elements" |
| `unsupported_multizone.cgns` | two zones | - | "multi-zone meshes are not supported yet" |
| `unsupported_structured.cgns` | a structured zone | - | "zone 'Block' is structured" |
| `invalid_name_type_conflict.cgns` | as `channel_sections` | BC `inlet` with BCType `BCWall` | "the name means inlet, but CGNS BCType BCWall means wall" |

The unsupported files are small ADF files. The files were checked with the CGNS project's validator `cgnscheck`
(Ubuntu package `cgns-convert`). The valid ones give only the two cosmetic warnings "dataclass not given" and, for
`CoordinateZ = 0`, "coordinate range is 0". The errors in the `unsupported_*` files are the intended defects.

## Files written by Gmsh

`square_gmsh.cgns`, `channel_cylinder_gmsh.cgns`: exported by Gmsh 4.15.2 from `../gmsh/square_4.1_ascii.msh` and
`../gmsh/channel_cylinder_4.1_binary.msh` with `export_gmsh_fixtures.py` (needs `pip install gmsh`).

Gmsh writes:
- every curve as a `FamilySpecified` BC named after the curve (`L_1`, `L_2`, ...), in a family of the same name
  without `FamilyBC`, on `PointRange` / `CellCenter` ranges of `BAR_2` elements;
- one "BC" for the surface (`S_1`, on the triangles), which is ignored with a warning;
- the vertices in a different order than the `.msh` file.

These files carry no usable names, so the tests use `*.bofl` entries by BC number. They then match the `.msh`
meshes position by position.
