#pragma once

// Test doubles for the linalg interfaces. Besides supporting the tests, they
// demonstrate that user-defined operators and preconditioners plug into the
// Krylov solvers without touching the library:
//
//   DenseOperator          LinearOperator backed by a dense matrix
//   DensePreconditioner    Preconditioner z = B r with a given dense B
//                          (not pointwise: exercises the generic solver path)
//   CountingOperator       Decorator counting operator applications
//   ForwardingPreconditioner
//                          Decorator counting applications; can hide the
//                          pointwise scaling to force the non-fused CG path
//
// Plus small dense model problems (1D Laplacian, indefinite matrix) and a
// few vector helpers.

#include "LinearSolverTestUtils.hpp"
#include "linalg/Linalg.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hydrama::test {

namespace la = hydrama::linalg;

class DenseOperator final : public la::LinearOperator {
public:
    explicit DenseOperator(DenseMatrix a) : a_(std::move(a)) {
        for (const auto& row : a_) {
            if (row.size() != a_.size()) throw std::invalid_argument("DenseOperator: matrix not square");
        }
    }
    std::size_t size() const override { return a_.size(); }
    void apply(la::ConstVectorView x, la::VectorView y) const override {
        for (std::size_t i = 0; i < a_.size(); ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < a_.size(); ++j) s += a_[i][j] * x[j];
            y[i] = s;
        }
    }
    void diagonal(la::VectorView d) const override {
        for (std::size_t i = 0; i < a_.size(); ++i) d[i] = a_[i][i];
    }
    la::OperatorTraits traits() const override {
        for (std::size_t i = 0; i < a_.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
                if (a_[i][j] != a_[j][i]) return {false};
        return {true};
    }
    std::string_view name() const override { return "dense"; }
    const DenseMatrix& matrix() const { return a_; }

private:
    DenseMatrix a_;
};

// Operator without a diagonal (uses the LinearOperator default).
class NoDiagonalOperator final : public la::LinearOperator {
public:
    explicit NoDiagonalOperator(std::size_t n) : n_(n) {}
    std::size_t size() const override { return n_; }
    void apply(la::ConstVectorView x, la::VectorView y) const override { la::copy(x, y); }
    std::string_view name() const override { return "no-diagonal"; }

private:
    std::size_t n_;
};

class DensePreconditioner final : public la::Preconditioner {
public:
    explicit DensePreconditioner(DenseMatrix b) : b_(std::move(b)) {}
    void setup(const la::LinearOperator& a) override {
        if (a.size() != b_.size()) throw std::invalid_argument("DensePreconditioner: size mismatch");
        n_ = a.size();
        ++setups;
    }
    void apply(la::ConstVectorView r, la::VectorView z) const override {
        for (std::size_t i = 0; i < n_; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < n_; ++j) s += b_[i][j] * r[j];
            z[i] = s;
        }
    }
    std::size_t size() const override { return n_; }
    std::string_view name() const override { return "dense"; }
    int setups = 0;

private:
    DenseMatrix b_;
    std::size_t n_ = 0;
};

class CountingOperator final : public la::LinearOperator {
public:
    explicit CountingOperator(const la::LinearOperator& a) : a_(a) {}
    std::size_t size() const override { return a_.size(); }
    void apply(la::ConstVectorView x, la::VectorView y) const override {
        ++applications;
        a_.apply(x, y);
    }
    void diagonal(la::VectorView d) const override { a_.diagonal(d); }
    la::OperatorTraits traits() const override { return a_.traits(); }
    std::string_view name() const override { return "counting"; }
    mutable int applications = 0;

private:
    const la::LinearOperator& a_;
};

class ForwardingPreconditioner final : public la::Preconditioner {
public:
    ForwardingPreconditioner(la::Preconditioner& inner, bool expose_scaling)
        : inner_(inner), expose_scaling_(expose_scaling) {}
    void setup(const la::LinearOperator& a) override { inner_.setup(a); }
    void apply(la::ConstVectorView r, la::VectorView z) const override {
        ++applications;
        inner_.apply(r, z);
    }
    std::size_t size() const override { return inner_.size(); }
    la::PreconditionerTraits traits() const override { return inner_.traits(); }
    std::string_view name() const override { return "forwarding"; }
    la::ConstVectorView pointwise_scaling() const override {
        return expose_scaling_ ? inner_.pointwise_scaling() : la::ConstVectorView{};
    }
    mutable int applications = 0;

private:
    la::Preconditioner& inner_;
    bool expose_scaling_;
};

// Records the order of the observer callbacks.
class EventLog final : public la::IterationMonitor {
public:
    void on_start(std::string_view, double) override { events.push_back("start"); }
    void on_iteration(int k, double) override { events.push_back("iter" + std::to_string(k)); }
    void on_finish(const la::SolveResult&) override { events.push_back("finish"); }
    std::vector<std::string> events;
};

// 1D Laplacian tridiag(-1, 2, -1) scaled by `scale`: SPD, condition ~ n^2.
inline DenseMatrix laplacian_1d(int n, double scale = 1.0) {
    DenseMatrix a = zeros(n);
    for (int i = 0; i < n; ++i) {
        a[i][i] = 2.0 * scale;
        if (i > 0) a[i][i - 1] = -scale;
        if (i + 1 < n) a[i][i + 1] = -scale;
    }
    return a;
}

// Diagonal matrix with the given entries.
inline DenseMatrix diagonal_matrix(const std::vector<double>& d) {
    DenseMatrix a = zeros(static_cast<int>(d.size()));
    for (size_t i = 0; i < d.size(); ++i) a[i][i] = d[i];
    return a;
}

// Dense inverse via column-wise direct solves.
inline DenseMatrix inverse(const DenseMatrix& a) {
    const int n = static_cast<int>(a.size());
    DenseMatrix inv = zeros(n);
    for (int j = 0; j < n; ++j) {
        std::vector<double> e(n, 0.0);
        e[j] = 1.0;
        const std::vector<double> col = solve_dense(a, e);
        for (int i = 0; i < n; ++i) inv[i][j] = col[i];
    }
    return inv;
}

// Deterministic, non-smooth test vector.
inline std::vector<double> test_vector(int n, double phase = 1.0) {
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = std::sin(3.7 * i + phase) + 0.5 * ((i * 13) % 7 - 3);
    return r;
}

inline double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

// ||b - A x|| over the rows not in `fixed`.
inline double residual_norm(const DenseMatrix& a, const std::vector<double>& b, const std::vector<double>& x,
                            const std::vector<int>& fixed = {}) {
    std::vector<double> r = multiply(a, x);
    for (size_t i = 0; i < r.size(); ++i) r[i] = b[i] - r[i];
    for (int d : fixed) r[d] = 0.0;
    return norm2(r);
}

// ---------------------------------------------------------------------------
// Sparse / assembled helpers
// ---------------------------------------------------------------------------

inline DenseMatrix to_dense(const la::CsrMatrix& a) {
    DenseMatrix d(a.rows(), std::vector<double>(a.cols(), 0.0));
    const auto& ptr = a.pattern().row_ptr();
    const auto& col = a.pattern().col_idx();
    for (int i = 0; i < a.rows(); ++i)
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) d[i][col[k]] = a.values()[k];
    return d;
}

// CSR matrix of the non-zero entries of a dense matrix (plus the full
// diagonal, which assembled preconditioners require to be stored).
inline la::CsrMatrix from_dense(const DenseMatrix& a, bool symmetric = false) {
    std::vector<la::Triplet> t;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[i].size(); ++j)
            if (a[i][j] != 0.0 || i == j) t.push_back({static_cast<int>(i), static_cast<int>(j), a[i][j]});
    la::CsrMatrix m = la::CsrMatrix::from_triplets(static_cast<int>(a.size()),
                                                   a.empty() ? 0 : static_cast<int>(a[0].size()), t);
    m.declare_symmetric(symmetric);
    return m;
}

// Kershaw (1978): SPD (eigenvalues 3 -+ 2 sqrt(2)), but not an M-matrix;
// IC(0) breaks down with a negative pivot (-5) in the last row.
inline DenseMatrix kershaw_matrix() {
    return {{3, -2, 0, 2}, {-2, 3, -2, 0}, {0, -2, 3, -2}, {2, 0, -2, 3}};
}

// Non-symmetric 2D convection-diffusion operator on an nx x ny grid
// (5-point diffusion + first-order upwind convection with velocity (bx, by)).
inline DenseMatrix convection_diffusion(int nx, int ny, double bx = 3.0, double by = 1.5) {
    const int n = nx * ny;
    DenseMatrix a = zeros(n);
    auto id = [nx](int i, int j) { return j * nx + i; };
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int r = id(i, j);
            a[r][r] = 4.0 + bx + by;
            if (i > 0) a[r][id(i - 1, j)] = -1.0 - bx;
            if (i + 1 < nx) a[r][id(i + 1, j)] = -1.0;
            if (j > 0) a[r][id(i, j - 1)] = -1.0 - by;
            if (j + 1 < ny) a[r][id(i, j + 1)] = -1.0;
        }
    }
    return a;
}

// Dense matrix of a linear map given by its action (column by column).
template <class Apply>
DenseMatrix matrix_of(int n, Apply apply) {
    DenseMatrix m = zeros(n);
    std::vector<double> e(n, 0.0), col(n);
    for (int j = 0; j < n; ++j) {
        e[j] = 1.0;
        apply(e, col);
        for (int i = 0; i < n; ++i) m[i][j] = col[i];
        e[j] = 0.0;
    }
    return m;
}

// Dense M^{-1} of a set-up preconditioner.
inline DenseMatrix inverse_preconditioner_matrix(const la::Preconditioner& m) {
    return matrix_of(static_cast<int>(m.size()), [&m](const std::vector<double>& x, std::vector<double>& y) {
        m.apply(x, y);
    });
}

inline DenseMatrix matmul(const DenseMatrix& a, const DenseMatrix& b) {
    const size_t n = a.size(), k = b.size(), m = b.empty() ? 0 : b[0].size();
    DenseMatrix c(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < n; ++i)
        for (size_t l = 0; l < k; ++l)
            for (size_t j = 0; j < m; ++j) c[i][j] += a[i][l] * b[l][j];
    return c;
}

inline double max_abs_diff(const DenseMatrix& a, const DenseMatrix& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[i].size(); ++j) d = std::max(d, std::abs(a[i][j] - b[i][j]));
    return d;
}

inline bool is_identity(const DenseMatrix& a, double tol) {
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[i].size(); ++j)
            if (std::abs(a[i][j] - (i == j ? 1.0 : 0.0)) > tol) return false;
    return true;
}

// Deterministic, platform-independent shuffle (Fisher-Yates on mt19937,
// whose output sequence is fixed by the standard; std::shuffle is not).
inline std::vector<int> shuffled_indices(int n, std::uint32_t seed) {
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    std::mt19937 gen(seed);
    for (int i = n - 1; i > 0; --i) {
        const int j = static_cast<int>(gen() % static_cast<std::uint32_t>(i + 1));
        std::swap(p[i], p[j]);
    }
    return p;
}

// Mesh with node numbers relabelled: new node k is old node perm[k].
inline Mesh renumbered(const Mesh& m, const std::vector<int>& new_to_old) {
    std::vector<int> old_to_new(m.npoin);
    for (int k = 0; k < m.npoin; ++k) old_to_new[new_to_old[k]] = k;
    Mesh r = m;
    for (int k = 0; k < m.npoin; ++k) {
        r.x[k] = m.x[new_to_old[k]];
        r.y[k] = m.y[new_to_old[k]];
    }
    for (auto& e : r.intma)
        for (int& ip : e) ip = old_to_new[ip];
    for (auto& side : r.iside) {
        side.node1 = old_to_new[side.node1];
        side.node2 = old_to_new[side.node2];
    }
    return r;
}

// ---------------------------------------------------------------------------
// Non-symmetric model problems
// ---------------------------------------------------------------------------

enum class Convection { Upwind, Central };

// -laplace(u) + (bx, by) . grad(u) on an nx x ny grid (unit mesh size,
// lexicographic numbering), sparse:
//   Upwind   first-order upwind convection: an M-matrix for any velocity
//   Central  central differences: 5-point Laplacian + skew-symmetric
//            convection part, so the symmetric part is SPD ("positive
//            real"); off-diagonals change sign for cell Peclet > 2
inline la::CsrMatrix convection_diffusion_csr(int nx, int ny, double bx, double by,
                                              Convection scheme = Convection::Upwind) {
    std::vector<la::Triplet> t;
    auto id = [nx](int i, int j) { return j * nx + i; };
    const bool up = scheme == Convection::Upwind;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int r = id(i, j);
            t.push_back({r, r, up ? 4.0 + bx + by : 4.0});
            if (i > 0) t.push_back({r, id(i - 1, j), up ? -1.0 - bx : -1.0 - 0.5 * bx});
            if (i + 1 < nx) t.push_back({r, id(i + 1, j), up ? -1.0 : -1.0 + 0.5 * bx});
            if (j > 0) t.push_back({r, id(i, j - 1), up ? -1.0 - by : -1.0 - 0.5 * by});
            if (j + 1 < ny) t.push_back({r, id(i, j + 1), up ? -1.0 : -1.0 + 0.5 * by});
        }
    }
    return la::CsrMatrix::from_triplets(nx * ny, nx * ny, t);
}

// A = S D S^{-1}: diagonalizable, non-symmetric, with the eigenvalues
// `eig` (S unit upper triangular). Column k of S is the eigenvector of eig[k].
struct SpectralMatrix {
    DenseMatrix a;
    DenseMatrix s;
};

inline SpectralMatrix matrix_with_spectrum(const std::vector<double>& eig) {
    const int n = static_cast<int>(eig.size());
    SpectralMatrix m;
    m.s = zeros(n);
    for (int i = 0; i < n; ++i) {
        m.s[i][i] = 1.0;
        for (int j = i + 1; j < n; ++j) m.s[i][j] = 0.3 * std::sin(1.0 + i + 2.0 * j);
    }
    DenseMatrix sd = m.s;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) sd[i][j] *= eig[j];
    m.a = matmul(sd, inverse(m.s));
    return m;
}

// BiCGSTAB breakdown fixtures (M = I, x0 = 0, shadow residual r~ = b),
// constructed symbolically (sympy) so that the quantity vanishes in exact
// arithmetic; in double precision it is ~1e-16 relative.
struct BreakdownCase {
    DenseMatrix a;
    std::vector<double> b;
};
// (r~, A r~) = 0 for every r~: pivot (alpha) breakdown in the first step.
inline BreakdownCase rotation_breakdown() { return {{{0, 1}, {-1, 0}}, {1, 2}}; }
// (t, s) = 0 in the first step: stagnation (omega = 0); ||s|| = 1.1785113019775793.
inline BreakdownCase omega_breakdown() { return {{{1, 0, 0}, {0, -1, 0}, {0, 0, 2}}, {1, 0.52223296786709351, 1}}; }
// (r~, r_1) = 0: Lanczos (rho) breakdown at the start of the second step.
inline BreakdownCase rho_breakdown() { return {{{2, 2, -2}, {0, 1, 1}, {1, 1, 4}}, {1, 1.2743072053293157, 1}}; }

} // namespace hydrama::test
