# Mesh Import

HydraMa2D reads Gmsh and SU2 meshes natively, and CGNS meshes when it is built with the CGNS library; the external
Fortran converter is no longer needed. This document covers:
- the case layout and the supported formats;
- the import pipeline and the guarantees it gives the solver;
- how to extend it (new readers, other element types).

Boundary names and conditions are covered in [BOUNDARY_NAMING.md](BOUNDARY_NAMING.md).

---

## 1. Using a Gmsh, SU2 or CGNS mesh

Put the mesh in `constant/<case>.msh` (Gmsh), `constant/<case>.su2` (SU2) or `constant/<case>.cgns` (CGNS) instead of
`constant/<case>.mesh`:

```
MyCase/
├── constant/
│   ├── mycase.msh      # Gmsh mesh (MSH 2.2 or 4.1, ASCII or binary)   -- or --
│   ├── mycase.su2      # SU2 native mesh (ASCII)                       -- or --
│   ├── mycase.cgns     # CGNS (HDF5 or ADF; needs a build with CGNS, section 4)
│   └── mycase.bofl     # optional: flag codes by physical tag / marker number / BC number (overrides names)
├── system/             # unchanged: .control, .physical, .solver, .turbulence
└── results/
```

```bash
./HydraMa2D MyCase            # case directory
./HydraMa2D MyCase/mycase     # or case path
```

- **Exactly one** of `<case>.mesh`, `<case>.msh`, `<case>.su2` and `<case>.cgns` must exist. If more than one does, the run
  stops with `AMBIGUOUS MESH`, so it is always clear which mesh a run used.
- **Legacy cases** (`<case>.mesh` + `<case>.bofl`) are read exactly as before: same reader, same messages, same results.

### Preparing the mesh in Gmsh

```
// channel.geo (excerpt)
Physical Curve("inlet")          = {4};
Physical Curve("outlet")         = {2};
Physical Curve("Symmetry Plane") = {1, 3};
Physical Curve("cylinder_wall")  = {5, 6, 7, 8};
Physical Surface("fluid")        = {1};

Mesh.MshFileVersion = 4.1;   // or 2.2
Mesh.Binary = 1;             // optional: binary is smaller and exact
```

```bash
gmsh -2 channel.geo -o constant/channel.msh
```

- Put **every boundary curve** in a physical curve group whose name contains a boundary keyword (`wall`, `inlet`,
  `outlet`, `symmetry`, ...).
- Use **linear elements** (`Mesh.ElementOrder = 1`, the default): 3-node triangles, 4-node quadrilaterals
  (`Recombine`, transfinite surfaces, `Field[...] = BoundaryLayer` with `Quads = 1`), or both in one mesh.
- Mesh in the **x-y plane** (z = const).
- Element orientation does not matter: clockwise triangles and quadrilaterals are reoriented. Node numbering does not
  matter either. Quadrilaterals must be strictly convex ([QUADRILATERAL_ELEMENTS.md](QUADRILATERAL_ELEMENTS.md)).

The run log shows what was imported (with quadrilaterals, the third line reads
`Mesh import: 318 triangles, 48 quadrilaterals, 240 nodes, 66 boundary sides`):

```
Mesh file: MyCase/constant/mycase.msh
Mesh format: Gmsh MSH 4.1 binary
Mesh import: 246 triangles, 151 nodes, 56 boundary sides
Mesh import: 16 duplicate element record(s) merged (elements in several physical groups)
Boundary 'inlet' (tag 1): inlet, flag 503, 4 sides [name]
Boundary 'outlet' (tag 2): outlet, flag 504, 4 sides [name]
Boundary 'Symmetry Plane' (tag 3): symmetry, flag 506, 32 sides [name]
Boundary 'cylinderWall' (tag 4): wall, flag 500, 16 sides [name]
Boundary 'noSlip' (tag 5): wall, flag 500, 16 sides [name]
```

Problems are reported together, one per line (up to 10, then a count), naming the physical group, element tag, node
tag and coordinates:

```
ERROR: MyCase/constant/mycase.msh: invalid mesh (2 problems)
  - physical group 'airfoil' (tag 3): boundary name 'airfoil' (normalised 'airfoil') contains no boundary keyword; ...
  - 4 boundary line elements are in no physical group, e.g. node 12 (1, 0.5) - node 13 (1, 0.75); ...
```

---

## 2. Gmsh formats

| Format | ASCII | Binary |
| :--- | :---: | :---: |
| MSH 4.1 | yes | yes: either byte order, 4- or 8-byte `size_t` |
| MSH 2.2 (also 2.0 / 2.1 headers) | yes | yes: either byte order |
| MSH 4.0, 3.x, 1.0 | rejected: re-save as 4.1 or 2.2 (`Mesh.MshFileVersion`) | |

| Section | Handling |
| :--- | :--- |
| `$MeshFormat` | Required, first. Version, file type, data size, and the endianness marker in binary files. |
| `$PhysicalNames` | Names of physical groups. Names may contain spaces, and a name may be unquoted. |
| `$Entities` (4.1) | Physical tags of each entity. An entity in k groups gives each of its elements k group memberships. |
| `$Nodes` | 2.2: tag x y z. 4.1: entity blocks; parametric coordinates are read and discarded. |
| `$Elements` | Kept: 2-node lines, 3-node triangles, 4-node quadrangles. Skipped: points. |
| `$PartitionedEntities` | Rejected: save the mesh unpartitioned. |
| anything else | Skipped (`$Periodic`, `$NodeData`, `$ElementData`, `$Comments`, `$InterpolationScheme`, ...), as the format requires. |

Rejected elements get a message naming the element and the fix:
- **Higher-order** elements (3-node lines, 6-node triangles, ...): set `Mesh.ElementOrder = 1`.
- **3D** elements: mesh with `gmsh -2`.

Robustness:
- The whole file is read into memory and parsed with one cursor, so ASCII and binary share the same logic.
- Every count is checked against the remaining file size before any allocation: a corrupt header cannot request
  gigabytes.
- Binary reads are bounds-checked.
- Numbers are parsed with `std::from_chars`: locale-independent, and NaN/Inf rejected.
- Error messages give `file: line N` (ASCII) or `file: byte N` (binary).
- The tests feed every truncation of the test files and thousands of randomly corrupted copies to the reader. The
  only outcomes are success or a `MeshImportError`, checked under ASan/UBSan.

---

## 3. SU2 native format

```
%                                   '%' starts a comment (whole line or after data)
NDIME= 2                            dimension (3 is rejected)
NELEM= 64                           element lines: VTK type, node indices, optional element index
5 0 1 10 0                            5 = triangle, 9 = quadrilateral
...
NPOIN= 45                           point lines: x y, optional index   ("NPOIN= n n_domain" accepted)
0.0 0.0 0
...
NMARK= 4                            boundary markers
MARKER_TAG= inlet                     name -> boundary condition (naming convention)
MARKER_ELEMS= 4                       boundary elements: 3 = line, node indices, optional index
3 9 0
...
```

| Part of the format | Handling |
| :--- | :--- |
| Keywords | `NDIME=`, `NELEM=`, `NPOIN=`, `NMARK=`, `MARKER_TAG=`, `MARKER_ELEMS=`. Any case, blanks around `=`; sections in any order. `NDIME=` must come first, and each section appears once. |
| `NDIME=` | Must be 2; `NDIME= 3` gives "3D mesh". |
| `NELEM=` | Exactly the declared number of element lines: `type n_1 ... n_k [index]`. Types: 5 triangle, 9 quadrilateral. Lines (3) and vertices (1) belong to the markers; 3D types (10, 12, 13, 14) give "3D element". |
| `NPOIN=` | Exactly the declared number of `x y [index]` lines (finite numbers); an optional second count (domain points) must not exceed the first. Points are numbered by position, as SU2 does. The optional index is ignored, with a warning if it differs from the position. |
| Node indices | 0-based, checked against `NPOIN=` even if `NPOIN=` comes later in the file. The error names the line of the element. |
| `NMARK=` | For each marker, `MARKER_TAG= name` (non-empty, unique), `MARKER_ELEMS= k`, then k lines of type 3. The i-th marker (1-based) becomes physical tag i, named by its tag. |
| `NZONE=` / `IZONE=` | `NZONE= 1` accepted; multi-zone meshes are rejected. |
| `NPERIODIC=` / `PERIODIC_INDEX=` | Legacy (SU2 ≤ 6) periodic transformations: parsed and skipped. A warning appears unless it is the identity block old SU2 versions always wrote. |
| `FFD_*` | Free-form deformation boxes (always at the end of the file) are ignored. |
| Anything else | Errors: an unknown keyword, a data line outside a section (count too small), a section interrupted by a keyword or by the end of the file (count too large). |

Section counts are checked against the remaining file size before anything is allocated.

Boundary conditions come from the **marker names**, with the same convention as Gmsh physical names
([BOUNDARY_NAMING.md](BOUNDARY_NAMING.md)). Classic SU2 names such as `airfoil` (no keyword) or `farfield` (not
implemented) are reported. A `*.bofl` file keyed by **marker number** (1 = first marker of the file) overrides names:

```
2
1 500      <- marker 1 ("airfoil") is a wall
2 504      <- marker 2 ("farfield") is a pressure outlet
```

The SU2 and Gmsh paths share everything after reading: tag remapping, orientation repair, boundary matching,
conditions, report and errors. An SU2 mesh exported by Gmsh imports to the same mesh as its `.msh` source.

In error messages, triangles and quadrilaterals are numbered by their 0-based position in `NELEM=`. Boundary lines are numbered by
their 0-based position among all marker elements of the file. Nodes use their SU2 point index.

---

## 4. CGNS (optional)

### 4.1 Building with CGNS

CGNS support needs the CGNS library (≥ 3.3) with HDF5. It is an **optional** feature: without the library HydraMa2D
builds, tests and runs exactly as before. The log banner then says `Mesh formats: HydraMa, Gmsh, SU2 (built without
CGNS)`, and reading a `.cgns` file gives the error "HydraMa2D was built without CGNS support".

| CMake option / hint | Meaning |
| :--- | :--- |
| `HYDRAMA2D_WITH_CGNS=AUTO` (default) | Use CGNS if found, otherwise build without it. |
| `HYDRAMA2D_WITH_CGNS=ON` | CGNS is required: configuration fails if it is not found. |
| `HYDRAMA2D_WITH_CGNS=OFF` | Never use CGNS. |
| `CGNS_ROOT` or `CMAKE_PREFIX_PATH` | Installation prefix of CGNS (`include/cgnslib.h`, `lib/libcgns.*`). |
| `CGNS_USE_STATIC_LIBS=ON` | Prefer `libcgns.a`; HDF5 is then found with FindHDF5 (hint `HDF5_ROOT`) and linked too. |

```bash
sudo apt install libcgns-dev                           # Debian / Ubuntu (CGNS 3.4 on Ubuntu 24.04)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# -- HydraMa2D CGNS mesh input: enabled (CGNS 3.4.0, shared: /usr/lib/x86_64-linux-gnu/libcgns.so)

cmake -S . -B build -DCGNS_ROOT=/opt/cgns-4.5 -DHYDRAMA2D_WITH_CGNS=ON       # a custom installation
make CGNS=1 CGNS_PREFIX=/opt/cgns-4.5                                        # Makefile build (see Makefile)
```

`cmake/FindCGNS.cmake` handles both generations:
- **CGNS 4.x** installs a CMake package (`cgns-config.cmake`). It is used only to locate the installation, because
  its static target links a target named `hdf5-shared` that the package does not define.
- **CGNS 3.x** installs no CMake package: `cgnslib.h` and `libcgns` are searched directly.

The version comes from `cgnslib.h`. HDF5 support is detected from `CG_BUILD_HDF5` (3.x) or `HDF5_LIBRARY` (4.x) in
`cgnsconfig.h`.

A **shared** libcgns loads HDF5 itself. A **static** one needs HDF5 on the link line, which FindCGNS adds.

Only `src/mesh/CgnsReader.cpp` includes `cgnslib.h`. It is compiled with `HYDRAMA2D_HAVE_CGNS=1` when CGNS is found,
and as a stub otherwise.

**Run-time library path.** If CGNS (or HDF5) is installed outside the system library directories, the dynamic
loader must find it when HydraMa2D runs, e.g. `LD_LIBRARY_PATH=/opt/cgns-4.5/lib`. CMake prints a reminder. The
tests set the path themselves: Catch2 `DL_PATHS` is used both for test discovery and for the tests.

### 4.2 What is read

| CGNS node | Handling |
| :--- | :--- |
| `CGNSBase_t` | Exactly one, with `CellDimension` 2; `PhysicalDimension` 2 or 3 (then z = const). |
| `Zone_t` | Exactly one, unstructured. Several zones ("multi-zone meshes are not supported yet: merge them") and structured zones are reported. |
| `GridCoordinates_t` | `CoordinateX`, `CoordinateY` (and `CoordinateZ`), read as double. Other coordinates (`CoordinateR`, ...) are reported. With several grids, the first is used (warning). |
| `Elements_t` | `TRI_3` and `QUAD_4` cells, `BAR_2` boundary edges, `NODE` (ignored) and `MIXED` sections of those. `MIXED` is read with `cg_poly_elements_read`, which works with CGNS 3.4 and 4.x files. Element ranges must not overlap. |
| `ZoneBC_t` / `BC_t` | The k-th BC is boundary group k (physical tag k). See 4.3. |
| `Family_t` / `FamilyBC_t` | Family names, and the BCType of the family for `FamilySpecified` BCs. |

Connectivity is read only for the element types that can be used. Every other section is reported by name, type and
element range, and unsupported elements inside `MIXED` sections are counted per type:

| Reported | Types |
| :--- | :--- |
| "3D elements; HydraMa2D needs a 2D mesh" | `TETRA_*`, `PYRA_*`, `PENTA_*`, `HEXA_*` (34 types), or `CellDimension` 3 |
| "higher-order elements; HydraMa2D needs linear elements (BAR_2, TRI_3, QUAD_4)" | `BAR_3..5`, `TRI_6..15`, `QUAD_8..25` (14 types) |
| "polygonal / polyhedral elements (NGON_n, NFACE_n) are not supported" | `NGON_n`, `NFACE_n` |
| "element type ... is not supported" | `ElementTypeNull`, `ElementTypeUserDefined`, unknown codes |

All problems are collected and reported together, in the same format as for the other formats.

### 4.3 Boundary conditions

**Point sets.** A BC defines boundary **edges**:

| Point set | Meaning |
| :--- | :--- |
| `ElementRange`, `ElementList` | `BAR_2` element numbers (CGNS 3.x writes these as the next row). |
| `PointRange`, `PointList` at `EdgeCenter`, `FaceCenter` or `CellCenter` | `BAR_2` element numbers. Gmsh writes `CellCenter`. |
| `PointRange`, `PointList` at `Vertex` | Vertex numbers. The BC gets every boundary edge with both end points in the list (a warning says how many). |

**BCs that are not boundaries are ignored with a warning:**
- BCs on 2D cells (Gmsh writes one for each surface, e.g. `S_1`);
- BCs on `NODE` elements;
- empty BCs.

**Errors** (each reported):
- elements that do not exist;
- BCs mixing edges and cells;
- `PointListDonor` and other donor point sets;
- `I/J/KFaceCenter` locations;
- vertex numbers out of range;
- `BCDegenerateLine` / `BCDegeneratePoint`.

**Names and types.** A CGNS BC has a name and a `BCType`. HydraMa2D combines them:
- **Name:** the BC name. If it has no boundary keyword, its family name is used when that one has a keyword
  (`BC_1` in family `Inflow` → `Inflow`).
- **Declared type:** the `BCType`. For `FamilySpecified` (and for BCs whose own type carries no information) the
  family's `FamilyBC` type is used.
- **Combination** follows the naming rules ([BOUNDARY_NAMING.md](BOUNDARY_NAMING.md), "Declared types"):
  - a name without keyword takes the declared type (`top` + `BCWall` → wall);
  - a specific name refines it (`hot_wall` + `BCWallViscousIsothermal` → wall_hot; `pressure_inlet` + `BCInflow` →
    pressure inlet, rejected as not implemented);
  - an ambiguous name is settled by it (`inlet_wall` + `BCWall` → wall);
  - contradictions are errors (`inlet` + `BCWall`).

The run log shows the source of each condition:

```
Boundary 'inlet' (tag 1): inlet, flag 503, 4 sides [name + declared type]
Boundary 'left' (tag 1): inlet, flag 503, 4 sides [declared type]
Boundary 'L_4' (tag 4): inlet, flag 503, 4 sides [*.bofl]
```

A `*.bofl` file keyed by **BC number** (the k-th `BC_t` of the zone is k) overrides both. This is the way to use
meshes whose BCs carry neither a keyword nor a type, such as Gmsh exports (`L_1`, `L_2`, ...; no `FamilyBC`).

### 4.4 Compatibility notes

- **Files written by CGNS 4.x cannot be read by CGNS 3.x** ("written with a more recent version of the CGNS
  library"). CGNS 4.x reads 3.x files. The test fixtures are therefore written with CGNS 3.4.
- **ADF and HDF5 files are both read.** The reader detects the type with `cg_is_cgns` before `cg_open`: CGNS 4.x no
  longer detects it once `cg_set_file_type` was called in the process.
- The model constants (`include/mesh/CgnsModel.hpp`) are checked against `cgnslib.h` with `static_assert` in every
  CGNS build. The tests also compare the element table with `cg_npe` for all 57 types.
- Tested with CGNS 3.4.0 / HDF5 1.10 (Ubuntu 24.04) and CGNS 4.5.0 / HDF5 2.2 (Debian), both shared and static.

---

## 5. Pipeline and guarantees

```
read_gmsh(path)                  src/mesh/GmshReader.cpp   file -> MeshData (syntax only)
read_su2(path)                   src/mesh/Su2Reader.cpp    file -> MeshData (syntax only)
read_cgns(path)                  src/mesh/CgnsReader.cpp   file -> CgnsFile (library layer)
                                 src/mesh/CgnsTranslate.cpp   CgnsFile -> MeshData (no library)
   |
build_mesh(data, options)        src/mesh/MeshBuilder.cpp  MeshData -> hydrama::Mesh (topology, geometry, BCs)
   |
Mesh                             same structure as the legacy reader produces
```

`MeshData` (`include/mesh/MeshData.hpp`) is format-neutral: original node tags and coordinates, elements with
original node tags and **one** physical tag each, and the physical names. An element in several physical groups
appears once per group. MSH 2.2 stores it that way, and the 4.1 reader expands its entity-based groups the same way.

`build_mesh` establishes the invariants that the solver (and `check_mesh`, `include/mesh/MeshCheck.hpp`) relies on.
They are the same ones the legacy converter produced, as `check_mesh` confirms on the bundled `hinterkante.mesh`.

| Step | What happens |
| :--- | :--- |
| Node tags | Arbitrary 64-bit tags (sparse, huge, zero or negative) are mapped to 0-based indices **in file order**. Duplicate tags are errors. |
| Unused nodes | Nodes of no triangle or quadrilateral (geometry points, stray nodes) are dropped. They would have zero lumped mass. |
| Cells | Triangles in file order, then quadrilaterals in file order (element e ≥ number of triangles is quadrilateral e − ntri). Duplicate records (same node set) are merged. Repeated nodes are errors. A triangle and a quadrilateral may share an edge (hybrid meshes). |
| Planarity | All used nodes must satisfy z = const to 1e-9 × the x-y extent. |
| Orientation (triangles) | Signed area > 0 is enforced: clockwise triangles swap their 2nd and 3rd node (counted in the report). Triangles with \|2A\| ≤ 1e-12 × (longest edge)² are degenerate and rejected. |
| Orientation (quadrilaterals) | det J > 0 everywhere on the element is enforced. det J of the bilinear map is affine in (ξ, η), so this holds iff it holds at the 4 corners, where det J is ¼ × the cross product of the two corner edges. All 4 negative (clockwise): the node order is reversed keeping the first node, (a, b, c, d) → (a, d, c, b) (counted in the report). Mixed signs: "is not convex or self-intersecting (its Jacobian determinant changes sign)". A corner value with \|det J\| ≤ ¼ × 1e-12 × (longest edge)²: "is degenerate (a corner angle of 0 or 180 degrees: zero Jacobian determinant)". See [QUADRILATERAL_ELEMENTS.md](QUADRILATERAL_ELEMENTS.md), section 4. |
| Conformity | Every edge belongs to one or two cells (triangles or quadrilaterals), and two cells traverse their common edge in opposite directions. Otherwise: "non-manifold" or "overlap" error. Hanging nodes are not conforming: the edges around them are open boundary edges, reported as "boundary edges have no line element". |
| Boundary sides | Every line element must be a mesh edge. A line on a boundary edge (one cell) becomes a boundary side, oriented counter-clockwise with respect to that cell (domain on the left), so `Mesh::compute_face_normals` gives **outward** normals, also around holes. `elem` = that cell. Lines on interior edges are ignored without a boundary condition and rejected with one. |
| Coverage | Every boundary edge must be covered by a line element in a physical group that resolves to a condition. |
| Conditions | Per boundary group: the `*.bofl` entry of its tag, else its name combined with the boundary type declared in the file, if any (CGNS BCType; [BOUNDARY_NAMING.md](BOUNDARY_NAMING.md)). A side in several groups needs agreeing conditions; `side_id` = smallest tag. Unimplemented types are rejected (`BuildOptions::allow_unimplemented_boundaries` keeps them with flag 0, for tools and tests). |
| Errors | Collected per stage and thrown together in one `MeshImportError`. |
| Groups | `Mesh::boundary_groups`: every boundary group with its tag, name and sides (a side in several groups is listed in each). Informational: used to select bodies for force monitors by name; the solver itself uses the flags. |

**File order is preserved** (nodes, triangles, quadrilaterals, boundary sides), and elements are not reordered or
rotated beyond the orientation fix. A mesh exported from a legacy case therefore imports to a bit-identical `Mesh`, and the solver
reproduces the legacy run bit for bit. The regression tests check this for every format variant.

---

### 5.1 Mesh quality and `--check-mesh`

After the geometry is computed, every run prints a quality summary per element type
(`include/mesh/MeshQuality.hpp`), for every mesh format including legacy meshes. For the level-1 inflation-layer
mesh of the cylinder benchmark ([CYLINDER_BENCHMARK.md](CYLINDER_BENCHMARK.md)):

```
Mesh quality: 3684 triangles: angles 33.1 - 99.9 deg, aspect ratio max 1.80 (mean 1.12), skewness max 0.45 (mean 0.08), scaled Jacobian min 0.63
Mesh quality: 192 quadrilaterals: angles 84.3 - 95.7 deg, aspect ratio max 9.95 (mean 7.01), skewness max 0.06 (mean 0.06), scaled Jacobian min 1.00
Mesh quality: equiangle skewness 3661 excellent, 215 good, 0 fair, 0 poor, 0 bad
Mesh quality: most skewed element: triangle (element index 7, 0-based), skewness 0.449, at (6.40763, 0.801367)
```

| Metric | Definition | Ideal |
| :--- | :--- | :--- |
| angles | smallest and largest interior corner angle | 60° (triangle), 90° (quadrilateral) |
| aspect ratio | longest / shortest edge ("edge ratio", Stimpson et al. 2007). Large values are intended in inflation layers | 1 |
| skewness | equiangle skewness max((θ_max − θ_e) / (180° − θ_e), (θ_e − θ_min) / θ_e), θ_e = 60° / 90° (Gambit, ANSYS Fluent, Cubit) | 0; 1 = degenerate |
| scaled Jacobian | min over the corners of sin θ, divided by sin 60° for triangles (Verdict) | 1; ≤ 0 = inverted |

Skewness classes: excellent < 0.25 ≤ good < 0.5 ≤ fair < 0.75 ≤ poor < 0.9 ≤ bad. Elements in the class *bad* give
a warning: they degrade the accuracy, and since the time step follows the smallest element height, also the cost.

`HydraMa2D --check-mesh <case>` reads the mesh of a case as a run would (any format, with `*.bofl`), checks the
solver's invariants (`check_mesh`) and the parabolic inlets, prints the boundary sides per condition and the quality
report, and exits with status 0 (usable) or 1. It reads nothing else and writes nothing:

```
$ HydraMa2D --check-mesh Hinterkante.dev.Re100
Mesh check: 17377 triangles, 0 quadrilaterals, 8956 nodes, 533 boundary sides: the solver's invariants hold
Mesh check: flag 500 (wall): 380 boundary sides
Mesh check: flag 503 (inlet): 102 boundary sides
Mesh check: flag 504 (outlet): 51 boundary sides
Mesh quality: 17377 triangles: angles 15.7 - 128.1 deg, aspect ratio max 3.69 (mean 1.25), skewness max 0.74 (mean 0.17), scaled Jacobian min 0.31
...
```

Sources: Stimpson, Ernst, Knupp, Pébay & Thompson, *The Verdict Geometric Quality Library*, Sandia report
SAND2007-1751 (2007); ANSYS Fluent User's Guide, "Mesh Quality" (equiangle skewness).

---

## 6. Code map

| File | Content |
| :--- | :--- |
| `include/mesh/MeshData.hpp` | `MeshData`, `RawElement`, `ElementType`, `MeshImportError` |
| `include/mesh/GmshReader.hpp`, `src/mesh/GmshReader.cpp` | `read_gmsh`, `parse_gmsh`, `gmsh_element_name` |
| `include/mesh/Su2Reader.hpp`, `src/mesh/Su2Reader.cpp` | `read_su2`, `parse_su2`, `su2_element_name` |
| `include/mesh/CgnsModel.hpp` | The CGNS structure that is read (`CgnsFile`), the CGNS standard codes, element table |
| `include/mesh/CgnsReader.hpp`, `src/mesh/CgnsReader.cpp` | `read_cgns_file` (only file using `cgnslib.h`; stub without CGNS), `cgns_support_available`, `cgns_library_version` |
| `src/mesh/CgnsTranslate.cpp` | `cgns_to_mesh_data`, `boundary_type_of_cgns_bc`, `read_cgns` (always compiled) |
| `src/mesh/Problems.hpp` | Private: collects problems and reports them together (builder, CGNS) |
| `cmake/FindCGNS.cmake` | Finds CGNS 3.x / 4.x, shared or static (+ HDF5), target `HydraMa2D::CGNS` |
| `include/mesh/MeshBuilder.hpp`, `src/mesh/MeshBuilder.cpp` | `build_mesh`, `BuildOptions`, `ImportReport`, `format_report` |
| `include/mesh/MeshCheck.hpp`, `src/mesh/MeshCheck.cpp` | `check_mesh`: independent check of the solver's mesh invariants |
| `include/mesh/MeshQuality.hpp`, `src/mesh/MeshQuality.cpp` | Element quality metrics, `assess_mesh_quality`, `format_quality_report` |
| `include/CaseMeshCheck.hpp`, `src/CaseMeshCheck.cpp` | `check_case_mesh`: `HydraMa2D --check-mesh` |
| `include/mesh/BoundaryNaming.hpp`, `src/mesh/BoundaryNaming.cpp` | Boundary types, aliases, `resolve_boundary_name` |
| `include/IO.hpp`, `src/IO.cpp` | `IO::locate_mesh_file`, `IO::read_case_mesh`, `IO::read_gmsh_mesh`, `IO::read_su2_mesh`, `IO::read_cgns_mesh`, `IO::read_boundary_flag_table` |
| `src/Solver.cpp` | `initialize()` reads the mesh with `IO::read_case_mesh` (legacy, Gmsh, SU2 or CGNS) |
| `src/main.cpp` | Case discovery (`.mesh` / `.msh` / `.su2` / `.cgns`), `--boundary-names`, banner with the mesh formats |

---

## 7. Tests

| File | Tag | Content |
| :--- | :--- | :--- |
| `tests/test_boundary_naming.cpp` | `[boundary_naming]` | Table consistency, every alias, normalisation, longest match, families, ambiguity, codes, docs table = code table |
| `tests/test_gmsh_reader.cpp` | `[gmsh][reader]` | The 4 variants give identical data; byte order, `size_t` = 4, CRLF, blanks, 64-bit tags, binary bytes that look like whitespace, unknown sections, 2.x headers, element tags, parametric nodes, every error; truncation and corruption sweeps |
| `tests/test_mesh_builder.cpp` | `[mesh_import]` | Tag remapping, unused nodes, reorientation, duplicates, holes, randomised numbering / orientation / line direction, interior lines, `*.bofl` precedence, unimplemented types, every invalid mesh, `check_mesh` |
| `tests/test_gmsh_fixtures.cpp` | `[gmsh][fixtures]` | Files written by Gmsh 4.15.2 in all variants: counts, patches, areas, outward normals, variant agreement, clockwise surface, parametric, rejected files |
| `tests/test_su2_reader.cpp` | `[su2][reader]` | The specification example; optional indices, section order, CRLF, comments, blanks, keyword case, `NZONE`/`IZONE`, `NPOIN= n n`, `NPERIODIC`, FFD trailer; index-mismatch warning; quadrilaterals, empty markers; every malformed header, count, element, point and marker, with line numbers; truncation and corruption sweeps |
| `tests/test_su2_import.cpp` | `[su2][mesh_import]` | Synthetic fixtures (regular channel: areas, markers, outward normals; clockwise repair = identical mesh; all format variants = identical mesh; `airfoil`/`farfield` with `*.bofl`; mixed quadrilaterals; 8 malformed files); Gmsh-exported SU2 = Gmsh source mesh; benchmark mesh through SU2 identical; **solver bit-identical to the legacy case**; case discovery; solver errors; the channel runs with correct boundary values |
| `tests/test_cgns_translate.cpp` | `[cgns][translate]` | No library needed (every build). Tables (57 element types, 26 BCTypes); every layout (sections / MIXED with and without offsets, 6 point-set / location variants, vertex BCs, clockwise, 3D coordinates, families) gives the SU2 channel exactly; name + BCType combination and rejections; `*.bofl` by BC number; ignored BCs; benchmark mesh; every unsupported element type; invalid MIXED, bases, zones, coordinates, sections, BCs; element numbers in messages |
| `tests/test_cgns_reader.cpp` | `[cgns][file]` | Build with / without CGNS; `cg_npe` = element table; round trips through HDF5 and ADF for every layout; fixtures (valid, rejected, Gmsh exports with `*.bofl`); file type detection; read errors; benchmark mesh and **solver bit-identical to the legacy case**; CGNS cases in the solver. Without CGNS: the "not available" path, file tests skipped |
| `tests/test_quad_mesh.cpp` | `[quad][mesh_import]` | Quadrilaterals through `build_mesh`: numbering after the triangles, clockwise repair, non-convex / self-intersecting / degenerate elements, hanging nodes, duplicates, all Gmsh variants; `Mesh` quantities of quadrilateral and hybrid meshes. Further quadrilateral fixtures are tested in the Gmsh, SU2 and CGNS test files (tag `[quad]`) |
| `tests/test_mesh_import_regression.cpp` | `[mesh_import][hinterkante]` | Benchmark mesh through 8 Gmsh variants = legacy mesh bit for bit; the **solver reproduces the legacy run bit for bit**; case discovery, `*.bofl` strictness, solver errors; the real-Gmsh channel runs with correct boundary values |

The fixtures in `tests/data/gmsh/` were written by the Gmsh SDK. To regenerate them:

```bash
pip install gmsh
cd tests/data/gmsh && python3 generate_fixtures.py
```

The CGNS fixtures in `tests/data/cgns/` are written by `generate_fixtures.cpp` there (target
`hydrama2d_cgns_fixtures`, built against CGNS 3.4), and by Gmsh (`export_gmsh_fixtures.py`). See the README there.

The SU2 fixtures in `tests/data/su2/` are synthetic. `generate_fixtures.py` there needs no dependencies and
regenerates them, including the malformed files. It also re-exports the `*_gmsh.su2` files if the gmsh module is
installed.

`tests/GmshTestWriter.hpp` and `tests/Su2TestWriter.hpp` write the synthetic cases (all variants, foreign byte order, CRLF, ...). Real Gmsh reads
its output, which was checked against the Gmsh 4.15.2 SDK for every variant Gmsh itself accepts.

---

## 8. Extending

**Other formats.** Write a reader that fills `MeshData`, as `Su2Reader.cpp` does (or, for a library-based format,
a model plus a library-free translation as for CGNS):
- coordinates with whatever tags the format uses;
- triangles, quadrilaterals and boundary lines with their original node ids;
- one physical tag per element record, with boundary names in `physical_names` under dimension 1;
- optionally a declared boundary type per group in `boundary_hints` (as the CGNS BCType).

Then call `build_mesh`. Remapping, orientation, boundary matching, conditions from names (same convention), error
reporting and the report all come for free. Add a branch to `IO::locate_mesh_file` and `Solver::initialize`. Run
`check_mesh` on the results in the tests.

**Other element types.** Triangles and bilinear quadrilaterals are supported
([QUADRILATERAL_ELEMENTS.md](QUADRILATERAL_ELEMENTS.md)). Quadrilaterals show what a new linear element type needs:
- storage in `Mesh` (as `Mesh::quads`; elements are numbered by type, and `Mesh::element_nodes(e)` hides the type
  from loops that only need the nodes);
- an element kernel (`include/numerics/QuadElement.hpp`) and its branch in Steps 1–3, the pressure operator (EBE
  blocks and CSR assembly), the field operations, the turbulence models and the VTK writer;
- an orientation rule that guarantees det J > 0 on the whole element (`build_mesh`), and its edges in the edge map
  of `build_mesh`;
- the element type in the readers (`ElementType`, the Gmsh / SU2 / CGNS type tables).
Higher-order elements need more: mid-side nodes change the node graph, the lumped mass and the boundary sides.
