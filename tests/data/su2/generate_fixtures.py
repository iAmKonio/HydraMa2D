#!/usr/bin/env python3
"""Regenerates the SU2 test fixtures (tests/test_su2_import.cpp).

Synthetic fixtures (pure Python, no dependencies) in this directory:

  channel_regular.su2            8 x 4 cells on [0,4] x [0,1], two triangles per cell
                                 with alternating diagonals, canonical SU2 layout
                                 (comments, point and element indices), markers
                                 inlet / outlet / upper_wall / lower_wall
  channel_regular_clockwise.su2  the same mesh, every triangle clockwise and every
                                 marker line reversed, no indices
  channel_regular_variants.su2   the same mesh with every optional / legacy part of
                                 the format: NZONE/IZONE header, NPOIN before NELEM,
                                 "NPOIN= n n", blanks around '=', tabs, inline
                                 comments, CRLF, legacy NPERIODIC block, FFD trailer
  annulus_airfoil_farfield.su2   O-grid ring 0.5 <= r <= 2 (a hole with a curved
                                 boundary) with the classic SU2 marker names
                                 "airfoil" and "farfield"
  mixed_tri_quad.su2             channel with a quadrilateral "inflation" layer on
                                 the lower wall and triangles elsewhere
  malformed/*.su2                one defect each (see MALFORMED below)

Fixtures exported by Gmsh (only if the gmsh module is installed):

  square_gmsh.su2, channel_cylinder_gmsh.su2, square_quads_gmsh.su2
      written by Gmsh from tests/data/gmsh/{square_4.1_ascii,
      channel_cylinder_4.1_binary, square_quads_2.2_ascii}.msh

    python3 generate_fixtures.py      # run from this directory
"""
import math
import os

NX, NY, LX, LY = 8, 4, 4.0, 1.0


def node(i, j):
    return j * (NX + 1) + i


def channel(quad_rows=0):
    pts = [(LX * i / NX, LY * j / NY) for j in range(NY + 1) for i in range(NX + 1)]
    elems = []
    for j in range(NY):
        for i in range(NX):
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            if j < quad_rows:
                elems.append((9, [a, b, c, d]))
            elif (i + j) % 2 == 0:
                elems += [(5, [a, b, c]), (5, [a, c, d])]
            else:
                elems += [(5, [a, b, d]), (5, [b, c, d])]
    markers = [
        ("inlet", [[node(0, j + 1), node(0, j)] for j in range(NY)]),
        ("outlet", [[node(NX, j), node(NX, j + 1)] for j in range(NY)]),
        ("upper_wall", [[node(i + 1, NY), node(i, NY)] for i in range(NX)]),
        ("lower_wall", [[node(i, 0), node(i + 1, 0)] for i in range(NX)]),
    ]
    return pts, elems, markers


def num(v):
    return repr(float(v))


def write(name, pts, elems, markers, point_index=True, element_index=True, comments=True):
    lines = []
    c = (lambda t: lines.extend(["%", "% " + t, "%"])) if comments else (lambda t: None)
    c("Problem dimension")
    lines.append("NDIME= 2")
    c("Inner element connectivity")
    lines.append("NELEM= %d" % len(elems))
    for k, (t, n) in enumerate(elems):
        lines.append(" ".join(map(str, [t] + n + ([k] if element_index else []))))
    c("Node coordinates")
    lines.append("NPOIN= %d" % len(pts))
    for k, (x, y) in enumerate(pts):
        lines.append(" ".join([num(x), num(y)] + ([str(k)] if point_index else [])))
    c("Boundary elements")
    lines.append("NMARK= %d" % len(markers))
    for tag, lns in markers:
        lines.append("MARKER_TAG= " + tag)
        lines.append("MARKER_ELEMS= %d" % len(lns))
        lines += ["3 %d %d" % (a, b) for a, b in lns]
    text = "\n".join(lines) + "\n"
    with open(name, "w", newline="") as f:
        f.write(text)
    return text


def variants(pts, elems, markers):
    out = ["% every optional and legacy part of the SU2 format", "NZONE= 1", "", "IZONE= 1",
           "NDIME =2   % blanks around '=' and an inline comment", "",
           "NPOIN= %d %d" % (len(pts), len(pts))]
    out += ["\t%s\t%s   " % (num(x), num(y)) for x, y in pts]            # no indices, tabs, trailing blanks
    out += ["NELEM=%d" % len(elems)]
    out += ["%d  %s  %% element %d" % (t, " ".join(map(str, n)), k) for k, (t, n) in enumerate(elems)]
    out += ["NMARK= %d" % len(markers)]
    for tag, lns in markers:
        out += ["  MARKER_TAG=   %s  " % tag, "MARKER_ELEMS= %d" % len(lns), "%"]
        out += ["3 %d %d %d" % (a, b, k) for k, (a, b) in enumerate(lns)]    # with an element index
    out += ["NPERIODIC= 1", "PERIODIC_INDEX= 0", "0 0 0", "0 0 0", "0 0 0",
            "FFD_NBOX= 0", "FFD_NLEVEL= 0", "FFD data that is not parsed"]
    with open("channel_regular_variants.su2", "w", newline="") as f:
        f.write("\r\n".join(out) + "\r\n")


def annulus(nt=24, nr=6, r0=0.5, r1=2.0):
    pts = []
    for j in range(nr + 1):
        r = r0 * (r1 / r0) ** (j / nr)                                       # geometric radial spacing
        pts += [(r * math.cos(2 * math.pi * i / nt), r * math.sin(2 * math.pi * i / nt)) for i in range(nt)]
    n = lambda i, j: j * nt + i % nt
    elems = []
    for j in range(nr):
        for i in range(nt):
            elems += [(5, [n(i, j), n(i + 1, j), n(i + 1, j + 1)]), (5, [n(i, j), n(i + 1, j + 1), n(i, j + 1)])]
    markers = [("airfoil", [[n(i, 0), n(i + 1, 0)] for i in range(nt)]),
               ("farfield", [[n(i, nr), n(i + 1, nr)] for i in range(nt)])]
    write("annulus_airfoil_farfield.su2", pts, elems, markers)


# name -> (description, function(text) -> text)
MALFORMED = {
    "truncated.su2": lambda t: t[: t.index("NPOIN=")] .replace("NELEM= 64", "NELEM= 70"),
    "point_index_out_of_range.su2": lambda t: t.replace("\n5 0 1 10 0\n", "\n5 0 1 45 0\n", 1),
    "ndime_3.su2": lambda t: t.replace("NDIME= 2", "NDIME= 3"),
    "marker_without_elems.su2": lambda t: t.replace("MARKER_ELEMS= 4\n3 9 0\n3 18 9\n3 27 18\n3 36 27\n", ""),
    "bad_coordinate.su2": lambda t: t.replace("\n0.5 0.0 1\n", "\n0.5 abc 1\n", 1),
    "triangle_in_marker.su2": lambda t: t.replace("MARKER_ELEMS= 4\n3 9 0\n", "MARKER_ELEMS= 4\n5 9 0 10\n", 1),
    "too_many_elements.su2": lambda t: t.replace("NELEM= 64", "NELEM= 63"),
    "quad_with_three_nodes.su2": lambda t: t.replace("\n5 0 1 10 0\n", "\n9 0 1 10\n", 1),
}


def gmsh_exports():
    try:
        import gmsh
    except ImportError:
        print("gmsh not installed: Gmsh-exported fixtures not regenerated")
        return
    src = os.path.join("..", "gmsh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    for msh, su2 in [("square_4.1_ascii.msh", "square_gmsh.su2"),
                     ("channel_cylinder_4.1_binary.msh", "channel_cylinder_gmsh.su2"),
                     ("square_quads_2.2_ascii.msh", "square_quads_gmsh.su2")]:
        gmsh.clear()
        gmsh.open(os.path.join(src, msh))
        gmsh.write(su2)
    print("Gmsh-exported fixtures written with gmsh", gmsh.option.getString("General.Version"))
    gmsh.finalize()


if __name__ == "__main__":
    pts, elems, markers = channel()
    text = write("channel_regular.su2", pts, elems, markers)
    cw = [(t, [n[0], n[2], n[1]]) for t, n in elems]
    rev = [(tag, [[b, a] for a, b in lns]) for tag, lns in markers]
    write("channel_regular_clockwise.su2", pts, cw, rev, point_index=False, element_index=False, comments=False)
    variants(pts, elems, markers)
    annulus()
    qp, qe, qm = channel(quad_rows=1)
    write("mixed_tri_quad.su2", qp, qe, qm)

    os.makedirs("malformed", exist_ok=True)
    plain = write("/dev/null" if os.name != "nt" else "nul", pts, elems, markers, comments=False)
    for name, make in MALFORMED.items():
        bad = make(plain)
        assert bad != plain, name
        with open(os.path.join("malformed", name), "w", newline="") as f:
            f.write(bad)
    gmsh_exports()
    print("synthetic SU2 fixtures written")
