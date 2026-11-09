#pragma once

// Dense-matrix helpers for testing the matrix-free linear algebra in
// LinearSolver.cpp on small meshes.

#include "LinearSolver.hpp"
#include "MeshTestUtils.hpp"

#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hydrama::test {

using DenseMatrix = std::vector<std::vector<double>>;

inline DenseMatrix zeros(int n) { return DenseMatrix(n, std::vector<double>(n, 0.0)); }

// Dense matrix of the production operator, column by column:
// A e_j computed with matrix_vector_mult().
inline DenseMatrix dense_operator(const Mesh& m,
                                  const std::vector<std::array<double, 3>>& gstif,
                                  const std::vector<double>& pdiag) {
    DenseMatrix a = zeros(m.npoin);
    std::vector<double> e(m.npoin, 0.0), col;
    for (int j = 0; j < m.npoin; ++j) {
        e[j] = 1.0;
        matrix_vector_mult(m, gstif, pdiag, e, col);
        for (int i = 0; i < m.npoin; ++i) a[i][j] = col[i];
        e[j] = 0.0;
    }
    return a;
}

// Independent reference: sum over elements of dt_e * K_e, assembled densely
// from the element Laplacian stored in Mesh (pdiagE / gstifE).
inline DenseMatrix dense_reference_laplacian(const Mesh& m, const std::vector<double>& delte) {
    DenseMatrix a = zeros(m.npoin);
    const int pair[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int ie = 0; ie < m.nelem; ++ie) {
        const auto& n = m.intma[ie];
        for (int i = 0; i < 3; ++i) a[n[i]][n[i]] += delte[ie] * m.pdiagE[ie][i];
        for (int k = 0; k < 3; ++k) {
            const int gi = n[pair[k][0]], gj = n[pair[k][1]];
            a[gi][gj] += delte[ie] * m.gstifE[ie][k];
            a[gj][gi] += delte[ie] * m.gstifE[ie][k];
        }
    }
    return a;
}

inline std::vector<double> multiply(const DenseMatrix& a, const std::vector<double>& x) {
    std::vector<double> y(a.size(), 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j) y[i] += a[i][j] * x[j];
    return y;
}

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline double norm2(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

inline double max_abs(const DenseMatrix& a) {
    double r = 0.0;
    for (const auto& row : a)
        for (double v : row) r = std::max(r, std::abs(v));
    return r;
}

// Gaussian elimination with partial pivoting.
inline std::vector<double> solve_dense(DenseMatrix a, std::vector<double> b) {
    const int n = static_cast<int>(b.size());
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(a[i][k]) > std::abs(a[piv][k])) piv = i;
        if (std::abs(a[piv][k]) < 1e-300) throw std::runtime_error("singular matrix");
        std::swap(a[k], a[piv]);
        std::swap(b[k], b[piv]);
        for (int i = k + 1; i < n; ++i) {
            const double f = a[i][k] / a[k][k];
            for (int j = k; j < n; ++j) a[i][j] -= f * a[k][j];
            b[i] -= f * b[k];
        }
    }
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= a[i][j] * x[j];
        x[i] = s / a[i][i];
    }
    return x;
}

// True if the (symmetric) matrix admits a Cholesky factorisation, i.e. is SPD.
inline bool cholesky_succeeds(const DenseMatrix& a) {
    const int n = static_cast<int>(a.size());
    DenseMatrix l = zeros(n);
    for (int j = 0; j < n; ++j) {
        double d = a[j][j];
        for (int k = 0; k < j; ++k) d -= l[j][k] * l[j][k];
        if (!(d > 0.0)) return false;
        l[j][j] = std::sqrt(d);
        for (int i = j + 1; i < n; ++i) {
            double s = a[i][j];
            for (int k = 0; k < j; ++k) s -= l[i][k] * l[j][k];
            l[i][j] = s / l[j][j];
        }
    }
    return true;
}

// Remove the rows and columns of the given (Dirichlet) nodes.
inline DenseMatrix reduced(const DenseMatrix& a, const std::set<int>& removed) {
    DenseMatrix r;
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        if (removed.count(i)) continue;
        std::vector<double> row;
        for (int j = 0; j < static_cast<int>(a.size()); ++j)
            if (!removed.count(j)) row.push_back(a[i][j]);
        r.push_back(row);
    }
    return r;
}

inline std::vector<int> boundary_node_list(const Mesh& m) {
    std::set<int> s;
    for (const auto& side : m.iside) {
        s.insert(side.node1);
        s.insert(side.node2);
    }
    return {s.begin(), s.end()};
}

// Everything the pressure solve needs, assembled like Solver::run() does:
// geometry -> update_lhs(delte) -> (pdiag, gstif).
struct PoissonSystem {
    Mesh mesh;
    std::vector<double> delte;
    std::vector<double> elcoe2;
    std::vector<double> pdiag;
    std::vector<std::array<double, 3>> gstif;
};

inline PoissonSystem make_poisson_system(Mesh mesh, std::vector<double> delte = {}) {
    PoissonSystem s;
    s.mesh = std::move(mesh);
    compute_all_geometry(s.mesh);
    s.delte = delte.empty() ? std::vector<double>(s.mesh.nelem, 1.0) : std::move(delte);
    update_lhs(s.mesh, s.delte, s.elcoe2, s.pdiag, s.gstif);
    return s;
}

} // namespace hydrama::test
