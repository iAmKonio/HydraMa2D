# SU2 test fixtures

Used by `tests/test_su2_import.cpp`. Regenerate with `python3 generate_fixtures.py` from this directory (pure
Python; the `*_gmsh.su2` files are re-exported only if the `gmsh` module is installed).

| File | Content | Expected |
| :--- | :--- | :--- |
| `channel_regular.su2` | 8 x 4 cells on [0,4] x [0,1], alternating diagonals, canonical SU2 layout (comments, point and element indices). Markers `inlet`, `outlet`, `upper_wall`, `lower_wall` | 64 triangles, 45 nodes, 24 sides, flags 503/504/500/500 |
| `channel_regular_clockwise.su2` | Same mesh, every triangle clockwise, every marker line reversed, no indices, no comments | all 64 reoriented; identical to `channel_regular` |
| `channel_regular_variants.su2` | Same mesh with `NZONE`/`IZONE`, NPOIN before NELEM, `NPOIN= n n`, blanks around `=`, tabs, inline comments, CRLF, marker element indices, legacy `NPERIODIC`, `FFD_*` trailer | identical to `channel_regular` |
| `annulus_airfoil_farfield.su2` | O-grid ring 0.5 <= r <= 2 with the classic markers `airfoil` and `farfield` | rejected (no keyword / not implemented); valid with `*.bofl` {1: 500, 2: 504} |
| `mixed_tri_quad.su2` | Channel with a quadrilateral layer on the lower wall | 48 triangles, 8 quadrilaterals, 45 nodes, 24 sides; the lower-wall sides belong to quadrilaterals |
| `square_gmsh.su2`, `channel_cylinder_gmsh.su2`, `square_quads_gmsh.su2` | Exported by Gmsh 4.15.2 from the meshes in `../gmsh/` | same mesh as the `.msh` source (21 quadrilaterals for `square_quads_gmsh`) |
| `malformed/truncated.su2` | NELEM= 70 but the file ends after 64 elements | `line 66: unexpected end of file` |
| `malformed/too_many_elements.su2` | NELEM= 63 for 64 element lines | `unexpected data line` |
| `malformed/point_index_out_of_range.su2` | An element references point 45 of 45 | `element references point 45, but NPOIN= 45` |
| `malformed/ndime_3.su2` | NDIME= 3 | `3D mesh` |
| `malformed/marker_without_elems.su2` | MARKER_TAG= without MARKER_ELEMS= | `expected 'MARKER_ELEMS= count'` |
| `malformed/bad_coordinate.su2` | A non-numeric coordinate | `expected finite coordinates` |
| `malformed/triangle_in_marker.su2` | A triangle among the boundary lines | `contains a triangle (type 5)` |
| `malformed/quad_with_three_nodes.su2` | A type-9 element with three nodes | `needs 4 node indices` |
