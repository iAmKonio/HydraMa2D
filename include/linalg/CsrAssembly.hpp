#pragma once

// Element-to-CSR assembly for linear triangles (P1) and bilinear
// quadrilaterals (Q1).
//
// P1CsrAssembler builds, once per mesh,
//   * the sparsity pattern of the node graph: (i, j) is stored iff nodes i
//     and j share an element (all four nodes of a quadrilateral are coupled),
//     plus the full diagonal (every node, even one without elements), and
//   * a scatter map: position(e, a, b) = position of the entry coupling the
//     local nodes a and b of element e in the CSR value array,
// so that (re-)assembling a matrix with new values (e.g. a new time step)
// is a plain O(nelem) loop without any searching (standard finite element
// practice, cf. Zienkiewicz, Taylor & Zhu 2013; Saad 2003, Sec. 2.3).
//
// Two assembly entry points:
//   assemble_ebe()        from the element-by-element storage of the
//                         pressure operator (gstif, pdiag from
//                         update_lhs(), layout Mesh::num_offdiag_blocks):
//                         gives exactly the matrix that EbeOperator applies
//                         (bit-identical entries)
//   add_element_matrix()       generic: adds a full 3x3 triangle element matrix
//   add_quad_element_matrix()  generic: adds a full 4x4 quadrilateral element matrix

#include "Mesh.hpp"
#include "linalg/CsrMatrix.hpp"

#include <array>
#include <memory>
#include <vector>

namespace hydrama::linalg {

class P1CsrAssembler {
public:
    using ElementMatrix = std::array<std::array<double, 3>, 3>;
    using QuadElementMatrix = std::array<std::array<double, 4>, 4>;

    // Throws std::invalid_argument for inconsistent meshes (connectivity
    // out of range).
    explicit P1CsrAssembler(const Mesh& mesh);

    const std::shared_ptr<const SparsityPattern>& pattern() const { return pattern_; }
    int nelem() const { return static_cast<int>(offset_.size()) - 1; }
    int npoin() const { return pattern_->rows(); }

    // Position of the entry coupling local nodes a and b of element e.
    int position(int e, int a, int b) const { return map_[offset_[e] + nodes_[e] * a + b]; }

    // All-zero matrix on the mesh pattern (declared symmetric: the caller
    // re-declares if it assembles a non-symmetric form).
    CsrMatrix create_matrix() const;

    // m := the matrix of the EBE storage: off-diagonal entries summed over
    // elements in element order, diagonal = pdiag. m must share pattern()
    // (i.e. come from create_matrix()).
    void assemble_ebe(const Mesh& mesh,
                      const std::vector<std::array<double, 3>>& gstif,
                      const std::vector<double>& pdiag,
                      CsrMatrix& m) const;

    // m += element matrix ke of triangle e (local node order of intma[e]);
    // m must share pattern().
    void add_element_matrix(CsrMatrix& m, int e, const ElementMatrix& ke) const;
    // m += element matrix ke of quadrilateral element e (local node order of
    // its entry in Mesh::quads).
    void add_quad_element_matrix(CsrMatrix& m, int e, const QuadElementMatrix& ke) const;

private:
    void require_pattern(const CsrMatrix& m, const char* who) const;

    std::shared_ptr<const SparsityPattern> pattern_;
    std::vector<int> offset_; // element e: map_[offset_[e] ..], nodes_[e]^2 entries
    std::vector<int> nodes_;  // nodes per element
    std::vector<int> map_;
    int num_triangles_ = 0;
};

} // namespace hydrama::linalg
