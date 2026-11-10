#include "linalg/CsrAssembly.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace hydrama::linalg {

P1CsrAssembler::P1CsrAssembler(const Mesh& mesh) {
    const int n = mesh.npoin;
    const int ne = mesh.nelem;
    if (n < 0 || ne < 0 || static_cast<int>(mesh.intma.size()) != ne) {
        throw std::invalid_argument("P1CsrAssembler: inconsistent mesh dimensions");
    }
    for (int e = 0; e < ne; ++e) {
        for (int a = 0; a < 3; ++a) {
            const int ip = mesh.intma[e][a];
            if (ip < 0 || ip >= n) {
                throw std::invalid_argument("P1CsrAssembler: element " + std::to_string(e) + " references node " +
                                            std::to_string(ip) + " out of range");
            }
        }
    }

    // 1. Candidate columns per row (with duplicates): the diagonal plus the
    //    two other nodes of every incident element.
    std::vector<int> count(n + 1, 0);
    for (int i = 0; i < n; ++i) count[i + 1] = 1;
    for (int e = 0; e < ne; ++e)
        for (int a = 0; a < 3; ++a) count[mesh.intma[e][a] + 1] += 2;
    for (int i = 0; i < n; ++i) count[i + 1] += count[i];

    std::vector<int> candidates(count[n]);
    std::vector<int> fill(count.begin(), count.end() - 1);
    for (int i = 0; i < n; ++i) candidates[fill[i]++] = i;
    for (int e = 0; e < ne; ++e) {
        const auto& c = mesh.intma[e];
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
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

    // 3. Element scatter map.
    map_.resize(ne);
    for (int e = 0; e < ne; ++e) {
        const auto& c = mesh.intma[e];
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) map_[e][3 * a + b] = pattern_->find(c[a], c[b]);
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
    if (mesh.nelem != nelem() || mesh.npoin != npoin() || static_cast<int>(gstif.size()) != nelem() ||
        static_cast<int>(pdiag.size()) != npoin()) {
        throw std::invalid_argument("P1CsrAssembler::assemble_ebe: coefficient arrays do not match the mesh");
    }
    std::vector<double>& v = m.values();
    std::fill(v.begin(), v.end(), 0.0);

    // Off-diagonal pairs (0,1), (0,2), (1,2), both triangles of the matrix.
    for (int e = 0; e < nelem(); ++e) {
        const auto& pos = map_[e];
        const auto& g = gstif[e];
        v[pos[1]] += g[0]; // (0,1)
        v[pos[3]] += g[0]; // (1,0)
        v[pos[2]] += g[1]; // (0,2)
        v[pos[6]] += g[1]; // (2,0)
        v[pos[5]] += g[2]; // (1,2)
        v[pos[7]] += g[2]; // (2,1)
    }
    const std::vector<int>& diag = pattern_->diagonal_positions();
    for (int i = 0; i < npoin(); ++i) v[diag[i]] = pdiag[i];
}

void P1CsrAssembler::add_element_matrix(CsrMatrix& m, int e, const ElementMatrix& ke) const {
    require_pattern(m, "add_element_matrix");
    if (e < 0 || e >= nelem()) throw std::out_of_range("P1CsrAssembler::add_element_matrix: element out of range");
    std::vector<double>& v = m.values();
    const auto& pos = map_[e];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) v[pos[3 * a + b]] += ke[a][b];
}

} // namespace hydrama::linalg
