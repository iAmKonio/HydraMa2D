#include "linalg/CsrAssembly.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace hydrama::linalg {

P1CsrAssembler::P1CsrAssembler(const Mesh& mesh) {
    const int n = mesh.npoin;
    const int ne = mesh.nelem;
    if (n < 0 || ne < 0 || mesh.num_triangles() + mesh.num_quads() != ne) {
        throw std::invalid_argument("P1CsrAssembler: inconsistent mesh dimensions");
    }
    num_triangles_ = mesh.num_triangles();
    for (int e = 0; e < ne; ++e) {
        for (int ip : mesh.element_nodes(e)) {
            if (ip < 0 || ip >= n) {
                throw std::invalid_argument("P1CsrAssembler: element " + std::to_string(e) + " references node " +
                                            std::to_string(ip) + " out of range");
            }
        }
    }

    // 1. Candidate columns per row (with duplicates): the diagonal plus the
    //    other nodes of every incident element.
    std::vector<int> count(n + 1, 0);
    for (int i = 0; i < n; ++i) count[i + 1] = 1;
    for (int e = 0; e < ne; ++e) {
        const ElementNodes c = mesh.element_nodes(e);
        for (int ip : c) count[ip + 1] += c.size() - 1;
    }
    for (int i = 0; i < n; ++i) count[i + 1] += count[i];

    std::vector<int> candidates(count[n]);
    std::vector<int> fill(count.begin(), count.end() - 1);
    for (int i = 0; i < n; ++i) candidates[fill[i]++] = i;
    for (int e = 0; e < ne; ++e) {
        const ElementNodes c = mesh.element_nodes(e);
        for (int a = 0; a < c.size(); ++a) {
            for (int b = 0; b < c.size(); ++b) {
                if (a != b) candidates[fill[c[a]]++] = c[b];
            }
        }
    }

    // 2. Sort and remove duplicates row by row.
    std::vector<int> row_ptr(n + 1, 0), col_idx;
    col_idx.reserve(candidates.size());
    for (int i = 0; i < n; ++i) {
        const auto first = candidates.begin() + count[i];
        const auto last = candidates.begin() + count[i + 1];
        std::sort(first, last);
        const auto end = std::unique(first, last);
        col_idx.insert(col_idx.end(), first, end);
        row_ptr[i + 1] = static_cast<int>(col_idx.size());
    }
    pattern_ = std::make_shared<const SparsityPattern>(n, n, std::move(row_ptr), std::move(col_idx));

    // 3. Element scatter map (n_e^2 positions per element).
    offset_.assign(static_cast<std::size_t>(ne) + 1, 0);
    nodes_.assign(static_cast<std::size_t>(ne), 0);
    for (int e = 0; e < ne; ++e) {
        nodes_[e] = mesh.nodes_per_element(e);
        offset_[e + 1] = offset_[e] + nodes_[e] * nodes_[e];
    }
    map_.resize(static_cast<std::size_t>(offset_[ne]));
    for (int e = 0; e < ne; ++e) {
        const ElementNodes c = mesh.element_nodes(e);
        for (int a = 0; a < c.size(); ++a) {
            for (int b = 0; b < c.size(); ++b) map_[offset_[e] + c.size() * a + b] = pattern_->find(c[a], c[b]);
        }
    }
}

CsrMatrix P1CsrAssembler::create_matrix() const {
    CsrMatrix m(pattern_);
    m.declare_symmetric(true);
    return m;
}

void P1CsrAssembler::require_pattern(const CsrMatrix& m, const char* who) const {
    // Pointer identity (O(1)): matrices must come from create_matrix() or
    // share its pattern.
    if (m.pattern_ptr() != pattern_) {
        throw std::invalid_argument(std::string("P1CsrAssembler::") + who +
                                    ": matrix does not share the assembler's sparsity pattern "
                                    "(create it with create_matrix())");
    }
}

void P1CsrAssembler::assemble_ebe(const Mesh& mesh,
                                  const std::vector<std::array<double, 3>>& gstif,
                                  const std::vector<double>& pdiag,
                                  CsrMatrix& m) const {
    require_pattern(m, "assemble_ebe");
    if (mesh.nelem != nelem() || mesh.npoin != npoin() || mesh.num_triangles() != num_triangles_ ||
        static_cast<int>(gstif.size()) != mesh.num_offdiag_blocks() || static_cast<int>(pdiag.size()) != npoin()) {
        throw std::invalid_argument("P1CsrAssembler::assemble_ebe: coefficient arrays do not match the mesh");
    }
    std::vector<double>& v = m.values();
    std::fill(v.begin(), v.end(), 0.0);

    // Triangles: off-diagonal pairs (0,1), (0,2), (1,2), both triangles of the matrix.
    for (int e = 0; e < num_triangles_; ++e) {
        const int* pos = &map_[offset_[e]];
        const auto& g = gstif[e];
        v[pos[1]] += g[0]; // (0,1)
        v[pos[3]] += g[0]; // (1,0)
        v[pos[2]] += g[1]; // (0,2)
        v[pos[6]] += g[1]; // (2,0)
        v[pos[5]] += g[2]; // (1,2)
        v[pos[7]] += g[2]; // (2,1)
    }
    // Quadrilaterals: blocks (0,1), (0,2), (0,3) and (1,2), (1,3), (2,3).
    for (int e = num_triangles_; e < nelem(); ++e) {
        const int* pos = &map_[offset_[e]];
        const int q = e - num_triangles_;
        const auto& b0 = gstif[num_triangles_ + 2 * q];
        const auto& b1 = gstif[num_triangles_ + 2 * q + 1];
        const double k[6] = {b0[0], b0[1], b0[2], b1[0], b1[1], b1[2]};
        for (int p = 0; p < 6; ++p) {
            const int a = q1::kPairs[p][0], b = q1::kPairs[p][1];
            v[pos[4 * a + b]] += k[p];
            v[pos[4 * b + a]] += k[p];
        }
    }
    const std::vector<int>& diag = pattern_->diagonal_positions();
    for (int i = 0; i < npoin(); ++i) v[diag[i]] = pdiag[i];
}

void P1CsrAssembler::add_element_matrix(CsrMatrix& m, int e, const ElementMatrix& ke) const {
    require_pattern(m, "add_element_matrix");
    if (e < 0 || e >= nelem()) throw std::out_of_range("P1CsrAssembler::add_element_matrix: element out of range");
    if (nodes_[e] != 3) throw std::invalid_argument("P1CsrAssembler::add_element_matrix: element is not a triangle");
    std::vector<double>& v = m.values();
    const int* pos = &map_[offset_[e]];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) v[pos[3 * a + b]] += ke[a][b];
}

void P1CsrAssembler::add_quad_element_matrix(CsrMatrix& m, int e, const QuadElementMatrix& ke) const {
    require_pattern(m, "add_quad_element_matrix");
    if (e < 0 || e >= nelem()) throw std::out_of_range("P1CsrAssembler::add_quad_element_matrix: element out of range");
    if (nodes_[e] != 4) throw std::invalid_argument("P1CsrAssembler::add_quad_element_matrix: element is not a quadrilateral");
    std::vector<double>& v = m.values();
    const int* pos = &map_[offset_[e]];
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) v[pos[4 * a + b]] += ke[a][b];
}

} // namespace hydrama::linalg
