# Boundary Naming Convention

How HydraMa2D turns boundary **names** into boundary **conditions**.

Names come from:
- the physical groups of a Gmsh mesh (`Physical Curve("inlet") = {...};`);
- the markers of an SU2 mesh (`MARKER_TAG= inlet`);
- the BCs of a CGNS mesh (`BC_t` name or family name).

CGNS files also **declare** a boundary type (`BCType`). It is combined with the name by the same rules (section 2b). Legacy `*.mesh` cases keep using numeric flag codes from the `*.bofl` file
(see [Legacy flag codes](#legacy-flag-codes-and-bofl-overrides)).

The single source of truth is the table in `src/mesh/BoundaryNaming.cpp`. This document lists the same table. The test
`docs/BOUNDARY_NAMING.md lists exactly the table of the code` (`tests/test_boundary_naming.cpp`) fails if the two
disagree. The solver prints the table:

```bash
./HydraMa2D --boundary-names
```

---

## 1. Quick reference

Name each boundary curve after its type. The keyword can appear anywhere in the name, in any case and with any
separator:

| You want | Name the physical curve group e.g. | Flag |
| :--- | :--- | :---: |
| no-slip wall | `wall`, `bottom_wall`, `cylinderWall`, `Wall 2` | 500 |
| velocity inlet (u = 1, v = 0) | `inlet`, `inlet_left`, `velocityInlet` | 503 |
| pressure outlet (p = 0) | `outlet`, `Outlet-1`, `pressure_outlet` | 504 |
| periodic / cyclic boundary pair | `periodic`, `cyclic`, `periodic_left` | 505 |
| symmetry / free slip (u·n = 0) | `symmetry`, `Symmetry Plane`, `slip_wall` | 506 |
| backward-facing-step inflow profile | `inlet_profile` | 507 |
| fully developed (parabolic) inflow, mean velocity 1 | `parabolic_inlet`, `inlet_parabolic`, `poiseuille_inlet` | 508 |
| wall translating with prescribed velocity (e.g. lid) | `moving_wall`, `sliding_wall`, `lid` | 509 |
| wall rotating with prescribed angular velocity | `rotating`, `rotating_wall`, `rotor` | 510 |
| wall tagged T = 1 / T = 0 | `wall_hot` / `wall_cold` | 501 / 502 |

The canonical keywords are `wall`, `inlet`, `outlet`, `symmetry`, `periodic`, `rotating`, `moving_wall`, `farfield`
and the others in the [tables below](#3-the-tables). Each has aliases (e.g. `no_slip`, `inflow`, `outflow`,
`symmetryPlane`, `cyclic`, `rotor`, `freestream`).

Types in the second table (`periodic`, `farfield`, ...) are **recognised but not implemented**. A mesh
that uses one is rejected with a clear message. It is never run with a wrong condition.

---

## 2. How a name is resolved

`resolve_boundary_name()` applies four rules:

1. **Normalise.**
   - Lower case.
   - Every character that is not a letter or digit becomes a separator `_`.
   - camelCase and letter/digit boundaries are split.
   - Repeated separators collapse, and leading/trailing ones are removed.

   | Name | Normalised | Tokens |
   | :--- | :--- | :--- |
   | `Velocity-Inlet 2` | `velocity_inlet_2` | velocity, inlet, 2 |
   | `noSlipWall` | `no_slip_wall` | no, slip, wall |
   | `symmetryPlane` | `symmetry_plane` | symmetry, plane |
   | `CHTInterface` | `cht_interface` | cht, interface |
   | `wall2` | `wall_2` | wall, 2 |
   | `Wall_T1` | `wall_t_1` | wall, t, 1 |

2. **Legacy codes.** A purely numeric name is a legacy flag code: `503` means inlet. Only the codes of the first
   table are accepted. Other codes (e.g. `511`) must go through the `*.bofl` file.

3. **Whole-token matches.** Every keyword and alias (normalised the same way) is searched for as a contiguous sequence
   of **whole tokens**:
   - `bottom_wall_2` contains `wall`.
   - `wallace`, `walled` and `inletwall` contain nothing.

   A match that lies inside a longer match is discarded:
   - `pressure_inlet` is a pressure inlet, not an `inlet`.
   - `no_slip_wall` is a `wall`, not a `slip_wall`.
   - `symmetry_axis` is an `axis`.

4. **Agreement.** All remaining matches must name the same type, with one exception: a **specific type beats the
   generic type of its family**. The families are:
   - `wall`: `wall_hot`, `wall_cold`, `rotating`, `moving_wall`, `baffle`.
   - `inlet`: `inlet_profile`, `inlet_parabolic`, `pressure_inlet`, `mass_flow_inlet`, `supersonic_inlet`.
   - `outlet`: `mass_flow_outlet`, `convective_outlet`, `supersonic_outlet`.
   - `symmetry`: `axis`.

   So `rotor_wall` is `rotating`, `lid_wall` is `moving_wall` and `hot_wall_left` is `wall_hot`. Anything else is
   **ambiguous** and rejected: `inlet_wall`, `periodic_wall`, `symmetry_wall`, `farfield_outlet`, `hot_wall_cold`.

Examples:

| Name | Result | Why |
| :--- | :--- | :--- |
| `wall`, `Walls`, `WALL_TOP`, `upper wall 2` | wall (500) | keyword/alias as a token |
| `cylinderWall`, `noSlip` | wall (500) | camelCase split |
| `Symmetry Plane`, `symmetryPlane`, `free_slip_wall` | symmetry (506) | alias `symmetry_plane`; `free_slip` and `slip_wall` agree |
| `velocityInlet`, `inlet-upper`, `inflow_1` | inlet (503) | |
| `pressure_outlet`, `static_pressure_outlet` | outlet (504) | longest match `pressure_outlet` |
| `pressure_inlet` | rejected: not implemented | longest match `pressure_inlet` |
| `supersonic_inlet` | rejected: not implemented | |
| `periodic_left`, `cyclicAMI` | periodic (505) | keyword periodic |
| `inlet_wall` | rejected: ambiguous | inlet and wall |
| `airfoil`, `left`, `boundary_1` | rejected: no keyword | rename, e.g. `airfoil_wall` |
| `503` | inlet (503) | legacy code |
| `505` | periodic (505) | legacy code |
| `511` | rejected: unknown code | use the `*.bofl` file |

---

## 2b. Declared types (CGNS BCType)

Some formats state the boundary type in addition to the name. This **declared type** is combined with the name using
the family rule of section 2, implemented by `resolve_boundary_name(name, hint)` and `combine_boundary_types`:

| Name | Declared type | Result |
| :--- | :--- | :--- |
| no keyword (`top`, `BC_3`, `L_1`), empty or an unknown code | T | T (`[declared type]` in the log) |
| resolves to T | T | T |
| resolves to a type S that refines T (`hot_wall`) | T (wall) | S (wall_hot) |
| resolves to T, and the declared type S refines T | S (axis) | S |
| ambiguous (`inlet_wall`) | T (wall) | the candidate compatible with T, if exactly one |
| resolves to T (`inlet`) | incompatible S (wall) | **error**: "the name means inlet, but CGNS BCType BCWall means wall" |

CGNS `BCType` → declared type (`boundary_type_of_cgns_bc`, `src/mesh/CgnsTranslate.cpp`):

| BCType | Declared type |
| :--- | :--- |
| `BCWall`, `BCWallViscous`, `BCWallViscousHeatFlux`, `BCWallViscousIsothermal` | `wall` |
| `BCWallInviscid`, `BCSymmetryPlane` | `symmetry` (slip: u·n = 0) |
| `BCSymmetryPolar` | `axis` (not implemented) |
| `BCInflow`, `BCInflowSubsonic`, `BCTunnelInflow` | `inlet` |
| `BCInflowSupersonic` | `supersonic_inlet` (not implemented) |
| `BCOutflow`, `BCOutflowSubsonic`, `BCTunnelOutflow` | `outlet` |
| `BCOutflowSupersonic` | `supersonic_outlet` (not implemented) |
| `BCFarfield` | `farfield` (not implemented) |
| `BCAxisymmetricWedge` | `wedge` (not implemented) |
| `BCTypeNull`, `BCTypeUserDefined`, `BCGeneral`, `BCDirichlet`, `BCNeumann`, `BCExtrapolate` | none: the name decides |
| `FamilySpecified` | the `FamilyBC` type of the BC's family (none if the family has no `FamilyBC`) |
| `BCDegenerateLine`, `BCDegeneratePoint` | error: not supported |

The generic inlet / outlet / wall types leave room for refinement by the name:
- `BCInflow` + `pressure_inlet` → pressure inlet, which is rejected;
- `BCInflow` + `inlet_profile` → the step inflow profile (507).

A `*.bofl` entry for the group overrides both the name and the declared type.

---

## 3. The tables

### Implemented by the solver

| Keyword | Flag | Aliases | Meaning |
| :--- | :---: | :--- | :--- |
| `wall` | 500 | `walls`, `no_slip`, `noslip`, `no_slip_wall`, `noslip_wall`, `viscous_wall`, `adiabatic_wall`, `solid_wall` | No-slip wall, u = v = 0 (adiabatic in the legacy code table). |
| `wall_hot` | 501 | `hot_wall`, `wall_t1`, `heated_wall`, `isothermal_hot` | No-slip wall tagged as constant temperature T = 1 (there is no energy equation: the flow sees a no-slip wall). |
| `wall_cold` | 502 | `cold_wall`, `wall_t0`, `cooled_wall`, `isothermal_cold` | No-slip wall tagged as constant temperature T = 0 (there is no energy equation: the flow sees a no-slip wall). |
| `inlet` | 503 | `inlets`, `inflow`, `velocity_inlet`, `inlet_velocity`, `velocity_inflow`, `uniform_inlet` | Velocity inlet, uniform u = 1, v = 0. |
| `outlet` | 504 | `outlets`, `outflow`, `pressure_outlet`, `outlet_pressure`, `exit`, `fixed_pressure` | Pressure outlet, p = 0 (velocity: natural / do-nothing condition). |
| `periodic` | 505 | `cyclic`, `cyclic_ami`, `periodicity` | Periodic / cyclic boundary pair. |
| `symmetry` | 506 | `sym`, `symm`, `symmetric`, `symmetry_plane`, `symmetry_line`, `mirror`, `slip`, `slip_wall`, `free_slip`, `freeslip`, `inviscid_wall`, `euler_wall` | Symmetry line / free-slip wall, u . n = 0. |
| `inlet_profile` | 507 | `profile_inlet`, `step_inlet`, `bfs_inlet`, `inlet_bfs` | Backward-facing-step inflow: fixed 6th-order polynomial u(y), v = 0. |
| `inlet_parabolic` | 508 | `parabolic_inlet`, `parabolic_inflow`, `poiseuille_inlet`, `inlet_poiseuille`, `fully_developed_inlet`, `developed_inlet` | Fully developed (Poiseuille) inflow: parabolic profile along the inward normal, mean velocity 1 over each connected inlet segment (1.5 at its centre). |
| `moving_wall` | 509 | `moving`, `sliding_wall`, `translating_wall`, `lid`, `moving_lid` | Wall translating with a prescribed tangential velocity (e.g. a cavity lid). |
| `rotating` | 510 | `rotating_wall`, `rotor`, `spinning_wall` | Wall rotating about an axis (prescribed angular velocity). |

### Recognised, not implemented (the mesh is rejected)

| Keyword | Flag | Aliases | Meaning |
| :--- | :---: | :--- | :--- |
| `farfield` | – | `far_field`, `freestream`, `free_stream` | Far-field / free-stream boundary. |
| `pressure_inlet` | – | `total_pressure_inlet`, `stagnation_inlet`, `stagnation_pressure_inlet`, `inlet_total_pressure`, `pressure_inflow` | Total (stagnation) pressure inlet. |
| `mass_flow_inlet` | – | `massflow_inlet`, `mass_flow_rate_inlet`, `mass_inflow` | Prescribed mass flow rate inlet. |
| `mass_flow_outlet` | – | `massflow_outlet`, `mass_flow_rate_outlet`, `mass_outflow` | Prescribed mass flow rate outlet. |
| `convective_outlet` | – | `advective_outlet`, `convective_outflow`, `advective_outflow` | Convective (advective) outflow condition. |
| `non_reflecting` | – | `nonreflecting`, `nrbc`, `riemann`, `giles`, `characteristic` | Non-reflecting / characteristic boundary (Riemann, Giles). |
| `opening` | – | `inlet_outlet`, `inletoutlet`, `entrainment`, `open_boundary` | Opening: inflow or outflow depending on the local flow direction. |
| `supersonic_inlet` | – | `supersonic_inflow` | Supersonic inflow (compressible flow only). |
| `supersonic_outlet` | – | `supersonic_outflow` | Supersonic outflow (compressible flow only). |
| `axis` | – | `axisymmetric`, `axisymmetry`, `symmetry_axis`, `axis_of_symmetry` | Axis of an axisymmetric domain (the solver is planar 2D). |
| `interface` | – | `interior`, `internal`, `fluid_interface`, `cht_interface`, `sliding_interface`, `mixing_plane`, `nearfield`, `near_field` | Internal interface between zones (fluid / conjugate heat transfer / sliding / mixing plane). |
| `baffle` | – | `baffles`, `thin_wall`, `internal_wall` | Zero-thickness internal wall. |
| `fan` | – | `actuator_disk`, `actuator_disc`, `intake_fan`, `exhaust_fan` | Fan / actuator disk: internal pressure-jump interface. |
| `porous_jump` | – | `radiator` | Porous jump / radiator: internal resistance interface. |
| `empty` | – | `front_and_back`, `frontandback`, `front_back` | Front / back planes of a one-cell-thick 3D mesh (not a boundary of a 2D mesh). |
| `wedge` | – | – | Wedge planes of an axisymmetric 3D mesh (not a boundary of a 2D mesh). |

**Why recognise types the solver cannot run?** Without them the token rule would quietly misread common names:
- `pressure_inlet`, `mass_flow_inlet` and `supersonic_inlet` would become a uniform velocity inlet.
- `convective_outlet` would become a p = 0 outlet.
- `rotating_wall` and `moving_wall` would become a fixed wall.

Recognising them turns each of these silent errors into an explicit "not implemented yet" message.

The internal type `custom` has no name. It stands for a `*.bofl` flag code outside the first table, which is kept
with a warning, as the legacy reader does.

---

## 4. What the conditions do in HydraMa2D

| Type | Velocity | Pressure | Turbulence (`BoundaryKind`) |
| :--- | :--- | :--- | :--- |
| `wall`, `wall_hot`, `wall_cold` | u = v = 0 at every node of the side | natural | Wall: wall distance, k = 0, model wall values |
| `inlet` | u = 1, v = 0 | natural | Inlet: inlet turbulence (`system/<case>.turbulence`) |
| `inlet_profile` | u = polynomial in y (fitted to the bundled backward-facing step, valid **only** for that geometry), v = 0 | natural | Inlet |
| `inlet_parabolic` | u = 6 s (L − s) / L² along the inward normal, s = arc length along the connected inlet segment of length L (mean 1, peak 1.5, 0 at the ends); any orientation, several segments allowed; closed or branching inlets are rejected at start-up | natural | Inlet |
| `outlet` | natural (do-nothing) | p = 0 (Dirichlet) | Outlet: zero gradient |
| `symmetry` | normal component removed, u·n = 0 | natural | Symmetry: zero gradient |

Notes:

- **No energy equation.** `wall_hot` and `wall_cold` keep their legacy codes 501/502 for a future energy equation.
  Today the flow sees an ordinary no-slip wall. Thermal qualifiers without a type of their own fall back to the token
  rule: `isothermal_wall` and `heat_flux_wall` are plain `wall`s.
- **Precedence at corners.** A node shared by sides of different types gets the condition applied last (legacy order):
  1. symmetry projection;
  2. `inlet_profile`;
  3. `inlet`;
  4. `inlet_parabolic`;
  5. walls.

  So walls always win, and an inlet corner on a symmetry line keeps u = 1, v = 0. An outlet node keeps p = 0 whatever
  its other sides are.
- **Pressure reference.** Without any `outlet` the pressure is fixed at the node `pnode` of `system/<case>.physical`.
- **Turbulence overrides** in `system/<case>.turbulence` are keyed by flag code, e.g. `boundary 506 symmetry`.

---

## 5. Physical groups in a Gmsh mesh

- **Boundary conditions come only from physical *curve* groups (dimension 1).** Surface groups (e.g. `fluid`) and
  point groups are ignored. Several surface groups all become part of the domain.
- **Every boundary edge must be in a physical curve group that resolves.** Untagged boundary edges are errors; the
  solver would otherwise treat them as unintended do-nothing boundaries.
- **A curve in several groups:** the groups must define the same condition, otherwise the mesh is rejected. The side
  gets the flag of that condition and, as `side_id`, the smallest tag of the groups. A group without a keyword (e.g.
  `all_boundaries`) is ignored with a warning if every one of its sides also belongs to a group that resolves.
- **Curves inside the domain** (embedded lines shared by two triangles) are ignored if their group has no boundary
  meaning (e.g. `refinement_line`). If their group resolves to a type (e.g. `wall`), the mesh is rejected: internal
  boundaries (baffles, interfaces) are not supported.
- **Physical group tags** become `BoundarySide::side_id`.

**CGNS BCs:**
- The k-th `BC_t` of the zone is boundary group k.
- Its name is the BC name, or its family name if only that one contains a keyword.
- Its declared type is the BCType (section 2b).
- Classic exports often have no usable names: Gmsh writes `L_1`, `L_2`, ... without types. Use a `*.bofl` file
  keyed by BC number for them.

**SU2 markers** follow the same rules:
- Marker i of the file (1-based, in `NMARK=` order) has physical tag i, and its `MARKER_TAG=` is the name.
- A side listed in several markers is in several groups, which must agree.
- Classic SU2 names need attention:
  - `airfoil` has no keyword: rename it (`airfoil_wall`) or use a `*.bofl` entry.
  - `farfield` is recognised but not implemented.
  - `lower_wall`, `upper_wall`, `inlet`, `outlet`, `symmetry` and `sym` resolve directly.
- Gmsh writes `Symmetry Plane` as `Symmetry_Plane` when exporting SU2. Both resolve to `symmetry`.

Example `.geo` file:

```
Physical Curve("inlet")          = {4};
Physical Curve("outlet")         = {2};
Physical Curve("Symmetry Plane") = {1, 3};
Physical Curve("cylinder_wall")  = {5, 6, 7, 8};
Physical Surface("fluid")        = {1};
```

---

## Legacy flag codes and `*.bofl` overrides

| Code | Type | Code | Type |
| :---: | :--- | :---: | :--- |
| 500 | `wall` | 504 | `outlet` |
| 501 | `wall_hot` | 505 | `periodic` |
| 502 | `wall_cold` | 506 | `symmetry` |
| 503 | `inlet` | 507 | `inlet_profile` |
| | | 508 | `inlet_parabolic` |
| | | 509 | `moving_wall` |
| | | 510 | `rotating` |
| | | 511 | unassigned (kept with a warning) |

- **Legacy cases** (`constant/<case>.mesh`) are unchanged. `constant/<case>.bofl` maps each side id to a code, and
  codes outside 500–508 give a warning. (Code 508 was unassigned before `inlet_parabolic`: a legacy `*.bofl` that used it as a private code now gets the parabolic inflow.)
- **Gmsh, SU2 and CGNS cases** (`constant/<case>.msh`, `.su2`, `.cgns`) may have a `constant/<case>.bofl` file in
  the same format. Gmsh entries are keyed by **physical curve tag**, SU2 entries by **marker number** (1 = first
  marker), CGNS entries by **BC number** (1 = first `BC_t` of the zone):

  ```
  2          <- number of entries
  4 507      <- physical tag 4: backward-facing-step inflow profile
  7 500      <- physical tag 7: wall
  ```

  An entry takes precedence over the group's name. Groups without an entry are resolved by name. This is the way to:
  - use unnamed groups (e.g. an old `.geo` file with `Physical Line(4) = {...}`);
  - keep a name that has no keyword (e.g. `airfoil`);
  - use a code without a keyword.

  The Gmsh, SU2 and CGNS paths check the file strictly: integer pairs, unique tags, and the declared count must be
  present.

---

## 6. Error messages and how to fix them

| Message (abridged) | Fix |
| :--- | :--- |
| `physical group 'airfoil' (tag 3): ... contains no boundary keyword` | Rename it (e.g. `airfoil_wall`) or add `3 500` to `constant/<case>.bofl` (for SU2: 3 = the third marker). |
| `physical group 'inlet_wall' (tag 2): ... is ambiguous: it matches inlet and wall` | Rename so that the name contains one boundary keyword. |
| `physical group 4 (unnamed): the group has no name` | Name the group, or add a `*.bofl` entry for tag 4. |
| `physical group 'farfield_1' (tag 5) is a 'farfield' boundary ..., which HydraMa2D does not implement yet` | Not yet possible. Choose an implemented type. |
| `N boundary line elements are in no physical group` | Add the curves to a named `Physical Curve`. |
| `N boundary edges have no line element` | Same. Since Gmsh saves only elements in physical groups, the curve is in none. |
| `... (wall) and in ... (outlet), which define different boundary conditions` | A curve is in two conflicting groups. Remove it from one. |
| `... the name means inlet, but CGNS BCType BCWall means wall` | The CGNS name and BCType disagree. Fix one of them, or add a `*.bofl` entry. |
| `... lies inside the domain ...; internal boundaries are not supported` | Remove the embedded curve from the boundary group. |
| `'511' ... is numeric but not a known flag code` | Use a keyword, or a `*.bofl` entry (legacy behaviour for unknown codes). |

---

## 7. For developers

**Adding an alias or a type that maps to an existing condition**
1. Add it to `make_table()` in `src/mesh/BoundaryNaming.cpp` and to the table in section 3.
2. Check it against the existing rules. An alias must normalise to a token sequence that no other type uses. The test
   `Keywords and aliases are unique and normalised` enforces this, and `docs/BOUNDARY_NAMING.md lists exactly the
   table of the code` enforces the docs.
3. Prefer distinctive words. Short generic words (`in`, `out`, `open`, `far`, `front`) would create false matches and
   ambiguities in names such as `hole_in_wall`.

**Implementing one of the recognised types (e.g. `periodic`)**
1. Give it a flag code. New codes must not clash with 500–508. Add the code to `flags::` and
   `classify_boundary_flag` in `include/BoundaryKind.hpp`.
2. Put the code in the type's table row (`flag`) and in section 3 of this document. `is_implemented()` then becomes
   true and meshes using the type are accepted.
3. Implement the condition in the solver (`src/BoundaryConditions.cpp`, and the turbulence boundary conditions if
   relevant), with tests.
4. Update `to_boundary_kind` expectations in `tests/test_boundary_naming.cpp` if the coarse kind changes.

**Where the names are used.** `build_mesh()` (`src/mesh/MeshBuilder.cpp`) resolves each physical curve group once:
first the `*.bofl` entry of its tag, then its name. It reports the result per group in the run log:

```
Boundary 'Symmetry Plane' (tag 3): symmetry, flag 506, 32 sides [name]
Boundary (unnamed) (tag 4): inlet_profile, flag 507, 51 sides [*.bofl]
```

Every reader (Gmsh physical names, SU2 marker tags, CGNS `BC_t` / family names) passes its names through
`resolve_boundary_name()` (via `build_mesh`), so the convention is the same for every format. Readers whose format
declares a type put it in `MeshData::boundary_hints`.
