// Pressure solver: solve_pcg() - element-by-element Jacobi-preconditioned
// conjugate gradients for A p = rhs, with A the assembled pressure operator
// (update_lhs) and Dirichlet values imposed at the nodes in bc_list.
//
// Convergence: stop when ||r|| / ||r_1|| < relToler or ||r|| < absToler, where
// ||r|| is the unpreconditioned residual 2-norm over the free nodes and r_1
// the residual after the FIRST iteration (see characterization test below).
// A tolerance of 0 switches that criterion off. Hard cap: 100000 iterations.

#include "IO.hpp"
#include "LinearSolverTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

struct Solve {
    PCGResult result;
    std::vector<double> press;
};

Solve run_pcg(const PoissonSystem& s, const std::vector<double>& rhs, const std::vector<int>& bc,
              const std::vector<double>& bc_values, double rel, double abs,
              std::vector<double> initial = {}) {
    Solve out;
    out.press = initial.empty() ? std::vector<double>(s.mesh.npoin, 0.0) : std::move(initial);
    out.result = solve_pcg(s.mesh, s.gstif, s.pdiag, rhs, bc, bc_values, rel, abs, out.press);
    return out;
}

// ||rhs - A x|| over the free (non-Dirichlet) nodes.
double true_residual(const PoissonSystem& s, const std::vector<double>& rhs,
                     const std::vector<int>& bc, const std::vector<double>& x) {
    std::vector<double> ax;
    matrix_vector_mult(s.mesh, s.gstif, s.pdiag, x, ax);
    std::vector<double> r(rhs.size());
    for (size_t i = 0; i < r.size(); ++i) r[i] = rhs[i] - ax[i];
    for (int ip : bc) r[ip] = 0.0;
    return norm2(r);
}

std::vector<double> values_at(const Mesh& m, const std::vector<int>& nodes, double (*f)(double, double)) {
    std::vector<double> v;
    for (int ip : nodes) v.push_back(f(m.x[ip], m.y[ip]));
    return v;
}

double linear_field(double x, double y) { return 2.0 - 1.5 * x + 0.75 * y; }

// Quadratic with Laplacian 2 - 1 = 1.
double quadratic_field(double x, double y) { return x * x - 0.5 * y * y + 0.3 * x * y + 0.2 * x; }

GridSpec pcg_grid() {
    GridSpec g;
    g.x0 = -0.5;
    g.y0 = 0.25;
    g.lx = 1.5;
    g.ly = 1.0;
    g.nx = 6;
    g.ny = 5;
    return g;
}

// Deterministic, non-smooth right-hand side.
std::vector<double> rough_rhs(int n) {
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = std::sin(3.7 * i + 1.0) + 0.5 * ((i * 13) % 7 - 3);
    return r;
}

} // namespace

// ---------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------

TEST_CASE("PCG reproduces a linear solution with Dirichlet data on the boundary", "[pcg]") {
    // Linear fields are discretely harmonic (A u = 0 at interior nodes), so
    // the discrete solution equals the exact one.
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const Mesh& m = s.mesh;
    const auto bc = boundary_node_list(m);
    const auto bc_values = values_at(m, bc, linear_field);

    const Solve sol = run_pcg(s, std::vector<double>(m.npoin, 0.0), bc, bc_values, 0.0, 1e-13);
    CHECK(sol.result.final_residual < 1e-13);
    for (int ip = 0; ip < m.npoin; ++ip) {
        CAPTURE(ip, m.x[ip], m.y[ip]);
        CHECK_THAT(sol.press[ip], WithinAbs(linear_field(m.x[ip], m.y[ip]), 1e-12));
    }
}

TEST_CASE("PCG reproduces a quadratic solution of -laplace(u) = f on a uniform grid", "[pcg]") {
    // On this triangulation A equals the 5-point stencil and the lumped mass
    // is hx*hy at interior nodes, so A u = M f holds exactly for quadratic u.
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const Mesh& m = s.mesh;
    const auto bc = boundary_node_list(m);
    const double f = -1.0; // -laplace(quadratic_field)

    std::vector<double> mass(m.npoin, 0.0), rhs(m.npoin);
    for (int ie = 0; ie < m.nelem; ++ie)
        for (int in = 0; in < 3; ++in) mass[m.intma[ie][in]] += m.elcoe_e[3 * ie + in];
    for (int ip = 0; ip < m.npoin; ++ip) rhs[ip] = mass[ip] * f;

    const Solve sol = run_pcg(s, rhs, bc, values_at(m, bc, quadratic_field), 0.0, 1e-13);
    for (int ip = 0; ip < m.npoin; ++ip) {
        CAPTURE(ip, m.x[ip], m.y[ip]);
        CHECK_THAT(sol.press[ip], WithinAbs(quadratic_field(m.x[ip], m.y[ip]), 1e-12));
    }
}

TEST_CASE("PCG agrees with a dense direct solve", "[pcg]") {
    // Single reference node as in the solver (pnode), element-wise time steps.
    const GridSpec g = pcg_grid();
    const PoissonSystem s = make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
    const int n = s.mesh.npoin;
    const int ref = GENERATE(0, 17);
    const double ref_value = 0.4;
    CAPTURE(ref);
    const std::vector<double> rhs = rough_rhs(n);

    // Dense reference: A_ff x_f = rhs_f - A_f,ref * ref_value.
    const DenseMatrix a = dense_operator(s.mesh, s.gstif, s.pdiag);
    std::vector<double> b;
    for (int i = 0; i < n; ++i)
        if (i != ref) b.push_back(rhs[i] - a[i][ref] * ref_value);
    const std::vector<double> xf = solve_dense(reduced(a, {ref}), b);

    const Solve sol = run_pcg(s, rhs, {ref}, {ref_value}, 0.0, 1e-13);
    CHECK(sol.press[ref] == ref_value);
    for (int i = 0, k = 0; i < n; ++i) {
        if (i == ref) continue;
        CAPTURE(i);
        CHECK_THAT(sol.press[i], WithinAbs(xf[k++], 1e-10));
    }
}

TEST_CASE("PCG imposes Dirichlet values exactly and ignores the initial guess there", "[pcg]") {
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const Mesh& m = s.mesh;
    const auto bc = boundary_node_list(m);
    const auto bc_values = values_at(m, bc, linear_field);

    const Solve sol = run_pcg(s, rough_rhs(m.npoin), bc, bc_values, 0.0, 1e-10,
                              std::vector<double>(m.npoin, 1.0e3));
    for (size_t k = 0; k < bc.size(); ++k) {
        CAPTURE(bc[k]);
        CHECK(sol.press[bc[k]] == bc_values[k]);
    }
}

TEST_CASE("PCG result does not depend on the initial guess", "[pcg]") {
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const int n = s.mesh.npoin;
    const std::vector<double> rhs = rough_rhs(n);

    const Solve a = run_pcg(s, rhs, {3}, {0.0}, 0.0, 1e-13);
    std::vector<double> guess(n);
    for (int i = 0; i < n; ++i) guess[i] = 5.0 * std::cos(0.3 * i);
    const Solve b = run_pcg(s, rhs, {3}, {0.0}, 0.0, 1e-13, guess);
    for (int i = 0; i < n; ++i) CHECK_THAT(b.press[i], WithinAbs(a.press[i], 1e-10));
}

TEST_CASE("PCG terminates within the number of unknowns", "[pcg]") {
    // In exact arithmetic CG converges in at most n_free iterations.
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const int n = s.mesh.npoin;

    SECTION("single reference node") {
        const Solve sol = run_pcg(s, rough_rhs(n), {0}, {0.0}, 0.0, 1e-10);
        CAPTURE(sol.result.iterations, n - 1);
        CHECK(sol.result.iterations <= n - 1);
    }
    SECTION("whole boundary fixed") {
        const auto bc = boundary_node_list(s.mesh);
        const Solve sol = run_pcg(s, rough_rhs(n), bc, std::vector<double>(bc.size(), 0.0), 0.0, 1e-10);
        const int n_free = n - static_cast<int>(bc.size());
        CAPTURE(sol.result.iterations, n_free);
        CHECK(sol.result.iterations <= n_free);
    }
}

TEST_CASE("PCG stops after one iteration when started from the solution", "[pcg]") {
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const Mesh& m = s.mesh;
    const auto bc = boundary_node_list(m);
    std::vector<double> exact(m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) exact[ip] = linear_field(m.x[ip], m.y[ip]);

    const Solve sol = run_pcg(s, std::vector<double>(m.npoin, 0.0), bc, values_at(m, bc, linear_field),
                              0.1, 1e-10, exact);
    CHECK(sol.result.iterations == 1);
    CHECK(sol.result.final_residual < 1e-12);
    for (int ip = 0; ip < m.npoin; ++ip) CHECK_THAT(sol.press[ip], WithinAbs(exact[ip], 1e-13));
}

TEST_CASE("PCG with zero data returns zero", "[pcg]") {
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const int n = s.mesh.npoin;
    const Solve sol = run_pcg(s, std::vector<double>(n, 0.0), {0}, {0.0}, 0.1, 1e-6);
    CHECK(sol.result.iterations == 1);
    CHECK(sol.result.final_residual == 0.0);
    for (double p : sol.press) CHECK(p == 0.0);
}

TEST_CASE("PCG first step is a Jacobi-preconditioned step", "[pcg]") {
    // From x0 = 0 the first iterate is x1 = alpha * z0 with z0 = D^-1 r0,
    // D = diag(A) and alpha = (r0.z0) / (z0.A z0). Element-wise time steps make
    // D strongly non-uniform, so an unpreconditioned step would differ.
    const GridSpec g = pcg_grid();
    const PoissonSystem s = make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
    const Mesh& m = s.mesh;
    const int n = m.npoin;
    const std::vector<double> rhs = rough_rhs(n);
    const int ref = 4;

    std::vector<double> r0 = rhs, z0(n), az;
    r0[ref] = 0.0;
    for (int i = 0; i < n; ++i) z0[i] = r0[i] / s.pdiag[i];
    matrix_vector_mult(m, s.gstif, s.pdiag, z0, az);
    az[ref] = 0.0;
    const double alpha = dot(r0, z0) / dot(z0, az);

    // A huge absolute tolerance stops the solver right after the first step.
    const Solve sol = run_pcg(s, rhs, {ref}, {0.0}, 0.0, 1e300);
    REQUIRE(sol.result.iterations == 1);
    for (int i = 0; i < n; ++i) {
        CAPTURE(i);
        CHECK_THAT(sol.press[i], WithinAbs(alpha * z0[i], 1e-12 * std::abs(alpha) * norm2(z0)));
    }
}

// ---------------------------------------------------------------------------
// Convergence control and reporting
// ---------------------------------------------------------------------------

TEST_CASE("PCG absolute tolerance", "[pcg]") {
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const int n = s.mesh.npoin;
    const std::vector<double> rhs = rough_rhs(n);
    const double abs_tol = GENERATE(1e-2, 1e-5, 1e-9);
    CAPTURE(abs_tol);

    const Solve sol = run_pcg(s, rhs, {0}, {0.0}, 0.0, abs_tol);
    CAPTURE(sol.result.iterations);
    CHECK(sol.result.final_residual < abs_tol);
    CHECK(true_residual(s, rhs, {0}, sol.press) < abs_tol * (1.0 + 1e-6));
}

TEST_CASE("PCG relative tolerance", "[pcg]") {
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const std::vector<double> rhs = rough_rhs(s.mesh.npoin);
    const double rel_tol = GENERATE(0.5, 0.1, 1e-4, 1e-8);
    CAPTURE(rel_tol);

    const Solve sol = run_pcg(s, rhs, {0}, {0.0}, rel_tol, 0.0);
    CAPTURE(sol.result.iterations, sol.result.initial_residual, sol.result.final_residual);
    CHECK(sol.result.initial_residual > 0.0);
    CHECK(sol.result.final_residual / sol.result.initial_residual < rel_tol);
}

TEST_CASE("PCG stops at whichever tolerance is met first", "[pcg]") {
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const std::vector<double> rhs = rough_rhs(s.mesh.npoin);
    const Solve rel_only = run_pcg(s, rhs, {0}, {0.0}, 1e-3, 0.0);
    const Solve abs_only = run_pcg(s, rhs, {0}, {0.0}, 0.0, 1e-9);
    const Solve both = run_pcg(s, rhs, {0}, {0.0}, 1e-3, 1e-9);
    REQUIRE(rel_only.result.iterations < abs_only.result.iterations);
    CHECK(both.result.iterations == rel_only.result.iterations);
}

TEST_CASE("Tighter tolerances cost more iterations and give smaller errors", "[pcg]") {
    const GridSpec g = pcg_grid();
    const PoissonSystem s = make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
    const int n = s.mesh.npoin;
    const std::vector<double> rhs = rough_rhs(n);
    const Solve exact = run_pcg(s, rhs, {0}, {0.0}, 0.0, 1e-14);

    int prev_iter = 0;
    double prev_err = 1e300;
    for (double tol : {1e-1, 1e-3, 1e-6, 1e-10}) {
        const Solve sol = run_pcg(s, rhs, {0}, {0.0}, 0.0, tol);
        double err = 0.0;
        for (int i = 0; i < n; ++i) err = std::max(err, std::abs(sol.press[i] - exact.press[i]));
        CAPTURE(tol, sol.result.iterations, err);
        CHECK(sol.result.iterations >= prev_iter);
        CHECK(err <= prev_err);
        prev_iter = sol.result.iterations;
        prev_err = err;
    }
    CHECK(prev_err < 1e-8);
}

TEST_CASE("PCG reports the true residual of the returned solution", "[pcg]") {
    // The recursively updated residual must match ||rhs - A x|| of the result.
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const std::vector<double> rhs = rough_rhs(s.mesh.npoin);
    const double abs_tol = GENERATE(1e-3, 1e-8);
    CAPTURE(abs_tol);

    const Solve sol = run_pcg(s, rhs, {0}, {0.0}, 0.0, abs_tol);
    CHECK_THAT(true_residual(s, rhs, {0}, sol.press),
               WithinAbs(sol.result.final_residual, 1e-12 * norm2(rhs)));
}

TEST_CASE("PCG initial residual is measured after the first iteration",
          "[pcg][characterization]") {
    // Documents current behaviour: PCGResult::initial_residual (printed by the
    // solver as "Initial residual") is ||r_1||, the residual AFTER the first
    // iteration, not ||r_0|| = ||rhs - A x_0||. The relative tolerance is
    // therefore measured against ||r_1|| as well.
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const Mesh& m = s.mesh;
    const int n = m.npoin;
    const std::vector<double> rhs = rough_rhs(n);
    const std::vector<int> bc = {0};

    // One Jacobi-PCG step, computed independently.
    std::vector<double> x0(n, 0.0), ax0, r0(n), z0(n), ap;
    matrix_vector_mult(m, s.gstif, s.pdiag, x0, ax0);
    for (int i = 0; i < n; ++i) r0[i] = rhs[i] - ax0[i];
    r0[0] = 0.0;
    for (int i = 0; i < n; ++i) z0[i] = r0[i] / s.pdiag[i];
    matrix_vector_mult(m, s.gstif, s.pdiag, z0, ap);
    ap[0] = 0.0;
    const double alpha = dot(r0, z0) / dot(z0, ap);
    std::vector<double> r1(n);
    for (int i = 0; i < n; ++i) r1[i] = r0[i] - alpha * ap[i];

    const Solve sol = run_pcg(s, rhs, bc, {0.0}, 1e-6, 0.0);
    CHECK_THAT(sol.result.initial_residual, WithinRel(norm2(r1), 1e-12));
    CHECK(std::abs(sol.result.initial_residual - norm2(r0)) > 1e-3 * norm2(r0));

    SECTION("so a relative tolerance above 1 stops after exactly one iteration") {
        const Solve one = run_pcg(s, rhs, bc, {0.0}, 1.5, 0.0);
        CHECK(one.result.iterations == 1);
        CHECK(one.result.final_residual == one.result.initial_residual);
    }
}

TEST_CASE("PCG with both tolerances switched off runs to the iteration cap",
          "[pcg][characterization]") {
    // Documents current behaviour: "0 = OFF" for both tolerances means there
    // is no stopping criterion left, and the solver always performs the
    // hard-coded maximum of 100000 iterations (PCGResult has no convergence
    // flag). The iterate must nevertheless stay finite and correct.
    const PoissonSystem s = make_poisson_system(make_grid_mesh(pcg_grid()));
    const Mesh& m = s.mesh;
    const auto bc = boundary_node_list(m);

    const Solve sol = run_pcg(s, std::vector<double>(m.npoin, 0.0), bc, values_at(m, bc, linear_field), 0.0, 0.0);
    CHECK(sol.result.iterations == 100000);
    for (int ip = 0; ip < m.npoin; ++ip) {
        CAPTURE(ip);
        REQUIRE(std::isfinite(sol.press[ip]));
        CHECK_THAT(sol.press[ip], WithinAbs(linear_field(m.x[ip], m.y[ip]), 1e-10));
    }
}

// ---------------------------------------------------------------------------
// Benchmark mesh
// ---------------------------------------------------------------------------

TEST_CASE("PCG on the Hinterkante mesh reproduces a linear solution", "[pcg][hinterkante]") {
    const std::string dir = HYDRAMA2D_CASE_DIR;
    Mesh mesh;
    REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", mesh));
    REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", mesh));
    const PoissonSystem s = make_poisson_system(std::move(mesh));
    const Mesh& m = s.mesh;
    const auto bc = boundary_node_list(m);

    const Solve sol = run_pcg(s, std::vector<double>(m.npoin, 0.0), bc, values_at(m, bc, linear_field), 0.0, 1e-11);
    CAPTURE(sol.result.iterations, sol.result.final_residual);
    CHECK(sol.result.final_residual < 1e-11);
    double worst = 0.0;
    for (int ip = 0; ip < m.npoin; ++ip)
        worst = std::max(worst, std::abs(sol.press[ip] - linear_field(m.x[ip], m.y[ip])));
    CAPTURE(worst);
    CHECK(worst < 1e-10);
}
