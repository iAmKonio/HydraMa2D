// PressureSolver: the facade through which Solver::run() solves the Step 2
// pressure Poisson equation (EBE operator + Jacobi + CG, legacy criteria).
//
// The main check replays a sequence of time steps (changing dt, global and
// local time stepping, warm-started pressure) and requires the facade, with
// its persistent workspace, to reproduce the frozen original PCG bit for bit
// at every step.

#include "BoundaryConditions.hpp"
#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "LinalgTestUtils.hpp"
#include "PressureSolver.hpp"
#include "reference/LegacyPcg.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace hydrama;
using namespace hydrama::test;

namespace {

SolverConfig solver_config(double rel, double abs, int solver_opt = 1) {
    SolverConfig c;
    c.solver_opt = solver_opt;
    c.relToler = rel;
    c.absToler = abs;
    return c;
}

Mesh benchmark_mesh() {
    Mesh mesh;
    CaptureStdout quiet;
    const std::string dir = HYDRAMA2D_CASE_DIR;
    REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", mesh));
    REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", mesh));
    compute_all_geometry(mesh);
    return mesh;
}

Mesh grid_mesh() {
    GridSpec g;
    g.lx = 2.0;
    g.ly = 1.0;
    g.nx = 10;
    g.ny = 5;
    Mesh m = make_grid_mesh(g);
    set_flags_by_side_id(m, {{1, 500}, {2, 504}, {3, 500}, {4, 503}});
    CaptureStdout quiet;
    compute_all_geometry(m);
    return m;
}

// Element time steps for step `k`: uniform (global time stepping) or
// varying per element (local time stepping).
std::vector<double> step_dt(int nelem, int k, bool local) {
    std::vector<double> dt(nelem);
    for (int ie = 0; ie < nelem; ++ie) {
        dt[ie] = 1e-3 * (1.0 + 0.1 * k);
        if (local) dt[ie] *= 1.0 + 0.5 * std::sin(0.7 * ie + k);
    }
    return dt;
}

} // namespace

TEST_CASE("PressureSolver reproduces the original PCG bit for bit over many time steps",
          "[pressure_solver][legacy]") {
    const bool real_mesh = GENERATE(false, true);
    const bool local_dt = GENERATE(false, true);
    const auto tol = GENERATE(std::pair<double, double>{0.1, 1e-6}, std::pair<double, double>{0.0, 1e-10});
    CAPTURE(real_mesh, local_dt, tol.first, tol.second);

    const Mesh mesh = real_mesh ? benchmark_mesh() : grid_mesh();
    const int n = mesh.npoin;
    PressureBC bc;
    {
        CaptureStdout quiet;
        bc = setup_pressure_bcs(mesh, 0);
    }
    REQUIRE(bc.nodes.size() > 1); // outlet nodes (504)

    PressureSolver solver(solver_config(tol.first, tol.second), bc);
    std::vector<double> elcoe2, pdiag, p(n, 1e-4), p_ref(n, 1e-4);
    std::vector<std::array<double, 3>> gstif;

    for (int k = 0; k < 4; ++k) {
        CAPTURE(k);
        update_lhs(mesh, step_dt(mesh.nelem, k, local_dt), elcoe2, pdiag, gstif);
        solver.update(mesh, gstif, pdiag);

        std::vector<double> rhs = test_vector(n, 0.3 * k);
        la::scale(rhs, 1e-4);

        const PCGResult ref = legacy::solve_pcg(mesh, gstif, pdiag, rhs, bc.nodes, bc.values, tol.first,
                                                tol.second, p_ref);
        const la::SolveResult res = solver.solve(rhs, p);
        CHECK(res.iterations == ref.iterations);
        CHECK(res.reference_residual == ref.initial_residual);
        CHECK(res.final_residual == ref.final_residual);
        CHECK(res.converged);
        REQUIRE(p == p_ref);
    }
}

TEST_CASE("PressureSolver is configured as the legacy PCG", "[pressure_solver]") {
    PressureBC bc;
    bc.nodes = {0, 3};
    bc.values = {0.0, 0.5};
    // solver_opt keeps its original meaning (time stepping only): any value
    // selects the PCG, exactly as before.
    const int solver_opt = GENERATE(1, 2);
    CAPTURE(solver_opt);
    const PressureSolver solver(solver_config(0.25, 1e-7, solver_opt), bc);

    CHECK(solver.krylov().name() == "CG");
    CHECK(solver.preconditioner().name() == "jacobi");
    const la::ConvergenceCriteria& c = solver.krylov().criteria();
    CHECK(c.rel_tol == 0.25);
    CHECK(c.abs_tol == 1e-7);
    CHECK(c.reference == la::ResidualReference::FirstIteration);
    CHECK(c.on_breakdown == la::BreakdownPolicy::Continue);
    CHECK(c.max_iterations == 100000);
    CHECK(solver.constraints().dofs() == bc.nodes);
    CHECK(solver.constraints().values() == bc.values);
    CHECK(solver.description() == "PCG (CG + jacobi, ebe)");
}

TEST_CASE("PressureSolver must be bound to an operator before solving", "[pressure_solver]") {
    const Mesh mesh = grid_mesh();
    PressureBC bc;
    bc.nodes = {0};
    bc.values = {0.0};
    PressureSolver solver(solver_config(0.1, 1e-6), bc);
    std::vector<double> rhs(mesh.npoin, 1.0), p(mesh.npoin, 0.0);
    CHECK_THROWS_AS(solver.solve(rhs, p), std::logic_error);

    std::vector<double> elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;
    update_lhs(mesh, std::vector<double>(mesh.nelem, 1e-2), elcoe2, pdiag, gstif);
    solver.update(mesh, gstif, pdiag);
    CHECK(solver.solve(rhs, p).converged);
}

TEST_CASE("PressureSolver rejects inconsistent input", "[pressure_solver]") {
    SECTION("constraint nodes and values differ in length") {
        PressureBC bc;
        bc.nodes = {0, 1};
        bc.values = {0.0};
        CHECK_THROWS_AS(PressureSolver(solver_config(0.1, 1e-6), bc), std::invalid_argument);
    }
    SECTION("constraint node outside the mesh") {
        const Mesh mesh = grid_mesh();
        PressureBC bc;
        bc.nodes = {mesh.npoin};
        bc.values = {0.0};
        PressureSolver solver(solver_config(0.1, 1e-6), bc);
        std::vector<double> elcoe2, pdiag;
        std::vector<std::array<double, 3>> gstif;
        update_lhs(mesh, std::vector<double>(mesh.nelem, 1e-2), elcoe2, pdiag, gstif);
        CHECK_THROWS_AS(solver.update(mesh, gstif, pdiag), std::out_of_range);
    }
}

TEST_CASE("PressureSolver re-binds to new coefficient arrays on update", "[pressure_solver]") {
    // The facade never keeps stale references: update() binds to whatever
    // arrays it is given (e.g. after the owning object moved).
    const Mesh mesh = grid_mesh();
    const int n = mesh.npoin;
    PressureBC bc;
    bc.nodes = {0};
    bc.values = {0.0};
    PressureSolver solver(solver_config(0.0, 1e-12), bc);

    std::vector<double> elcoe2, pdiag_a, pdiag_b;
    std::vector<std::array<double, 3>> gstif_a, gstif_b;
    update_lhs(mesh, std::vector<double>(mesh.nelem, 1e-2), elcoe2, pdiag_a, gstif_a);
    update_lhs(mesh, std::vector<double>(mesh.nelem, 4e-2), elcoe2, pdiag_b, gstif_b);
    const std::vector<double> rhs = test_vector(n);

    solver.update(mesh, gstif_a, pdiag_a);
    std::vector<double> pa(n, 0.0);
    solver.solve(rhs, pa);
    solver.update(mesh, gstif_b, pdiag_b);
    std::vector<double> pb(n, 0.0);
    solver.solve(rhs, pb);

    // A scales linearly with the uniform dt, so p_b = p_a / 4.
    for (int i = 0; i < n; ++i) {
        CAPTURE(i);
        CHECK(std::abs(pb[i] - 0.25 * pa[i]) <= 1e-9 * (1.0 + std::abs(pa[i])));
    }
}

TEST_CASE("PressureSolver forwards its monitor to the Krylov solver", "[pressure_solver][monitor]") {
    const Mesh mesh = grid_mesh();
    PressureBC bc;
    bc.nodes = {0};
    bc.values = {0.0};
    PressureSolver solver(solver_config(0.01, 0.0), bc);
    std::vector<double> elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;
    update_lhs(mesh, std::vector<double>(mesh.nelem, 1e-2), elcoe2, pdiag, gstif);
    solver.update(mesh, gstif, pdiag);

    la::ResidualHistory history;
    solver.set_monitor(&history);
    std::vector<double> p(mesh.npoin, 0.0);
    const la::SolveResult r = solver.solve(test_vector(mesh.npoin), p);
    CHECK(history.solves() == 1);
    CHECK(history.residuals().size() == static_cast<size_t>(r.iterations) + 1);

    solver.set_monitor(nullptr);
    solver.solve(test_vector(mesh.npoin), p);
    CHECK(history.solves() == 1);
}

// ---------------------------------------------------------------------------
// Configurable solver: CSR operator, assembled preconditioners, reordering
// ---------------------------------------------------------------------------

namespace {

SolverConfig configured(const std::string& preconditioner, const std::string& reordering, const std::string& format,
                        double rel = 1e-11, double abs = 0.0, const std::string& method = "cg") {
    SolverConfig c = solver_config(rel, abs);
    c.preconditioner = preconditioner;
    c.reordering = reordering;
    c.operator_format = format;
    c.method = method;
    return c;
}

} // namespace

TEST_CASE("Every solver configuration converges to the same pressure over many time steps",
          "[pressure_solver][configurations]") {
    const bool real_mesh = GENERATE(false, true);
    const bool local_dt = GENERATE(false, true);
    CAPTURE(real_mesh, local_dt);

    const Mesh mesh = real_mesh ? benchmark_mesh() : grid_mesh();
    const int n = mesh.npoin;
    PressureBC bc;
    {
        CaptureStdout quiet;
        bc = setup_pressure_bcs(mesh, 0);
    }

    // GMRES runs with a long restart: on this ill-conditioned SPD system
    // restarted GMRES(30) needs ~10x the iterations of CG (restarting
    // discards the Krylov space), GMRES(200) about the same as CG.
    struct Variant {
        std::string preconditioner, reordering, format, method = "cg";
        int restart = 30;
    };
    const std::vector<Variant> variants = {
        {"none", "none", "csr"},  {"jacobi", "none", "csr"}, {"ssor", "none", "ebe"}, {"ssor", "rcm", "csr"},
        {"ic0", "none", "ebe"},   {"ic0", "rcm", "ebe"},     {"ic0", "rcm", "csr"},   {"ilu0", "none", "csr"},
        {"ilu0", "rcm", "ebe"},   {"ilu0", "rcm", "csr", "bicgstab"}, {"jacobi", "none", "ebe", "bicgstab"},
        {"ilu0", "rcm", "csr", "gmres", 200}, {"ic0", "rcm", "ebe", "gmres", 200},
    };

    std::vector<std::unique_ptr<PressureSolver>> solvers;
    std::vector<std::vector<double>> pressures;
    for (const Variant& v : variants) {
        // BiCGSTAB/GMRES verify their residual against the true residual,
        // whose rounding floor is ~1e-13 relative here: they are run to
        // 1e-12 (CG reports its recursive residual, which keeps decreasing).
        const double rel = v.method == "cg" ? 1e-13 : 1e-12;
        SolverConfig c = configured(v.preconditioner, v.reordering, v.format, rel, 0.0, v.method);
        c.gmres_restart = v.restart;
        solvers.push_back(std::make_unique<PressureSolver>(c, bc));
        pressures.emplace_back(n, 1e-4);
    }
    std::vector<double> p_ref(n, 1e-4);

    std::vector<double> elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;
    for (int k = 0; k < 2; ++k) { // setup, then the update path
        CAPTURE(k);
        update_lhs(mesh, step_dt(mesh.nelem, k, local_dt), elcoe2, pdiag, gstif);
        std::vector<double> rhs = test_vector(n, 0.3 * k);
        la::scale(rhs, 1e-4);
        // Reference: the original PCG, converged as far as the variants. With
        // cond(A) ~ 1e5 a relative residual of 1e-13 fixes p to ~1e-8.
        legacy::solve_pcg(mesh, gstif, pdiag, rhs, bc.nodes, bc.values, 1e-13, 0.0, p_ref);
        double scale = 0.0;
        for (double v : p_ref) scale = std::max(scale, std::abs(v));

        for (size_t s = 0; s < variants.size(); ++s) {
            CAPTURE(solvers[s]->description());
            solvers[s]->update(mesh, gstif, pdiag);
            const la::SolveResult r = solvers[s]->solve(rhs, pressures[s]);
            CHECK(r.converged);
            CHECK(r.reason == la::StopReason::RelativeTolerance);
            CHECK(max_abs_diff(pressures[s], p_ref) < 1e-8 * scale);
            for (int d : bc.nodes) CHECK(pressures[s][d] == 0.0);
        }
    }
}

TEST_CASE("PressureSolver composition follows the configuration", "[pressure_solver][configurations]") {
    const Mesh mesh = grid_mesh();
    PressureBC bc;
    bc.nodes = {0};
    bc.values = {0.0};
    std::vector<double> elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;
    update_lhs(mesh, std::vector<double>(mesh.nelem, 1e-2), elcoe2, pdiag, gstif);

    SECTION("legacy: matrix-free, nothing assembled") {
        PressureSolver s(solver_config(0.1, 1e-6), bc);
        s.update(mesh, gstif, pdiag);
        CHECK(s.legacy());
        CHECK(s.op().name() == "ebe");
        CHECK(s.assembled_matrix() == nullptr);
        CHECK(s.constrained_matrix() == nullptr);
    }
    SECTION("CSR operator with Jacobi: assembled operator, no constrained copy") {
        PressureSolver s(configured("jacobi", "none", "csr"), bc);
        s.update(mesh, gstif, pdiag);
        CHECK_FALSE(s.legacy());
        CHECK(s.op().name() == "csr");
        REQUIRE(s.assembled_matrix() != nullptr);
        CHECK(&s.op() == s.assembled_matrix());
        CHECK(s.constrained_matrix() == nullptr);
        CHECK(s.description() == "PCG (CG + jacobi, csr)");
        CHECK(s.krylov().criteria().reference == la::ResidualReference::InitialResidual);
    }
    SECTION("EBE operator with an assembled preconditioner") {
        PressureSolver s(configured("ic0", "rcm", "ebe"), bc);
        s.update(mesh, gstif, pdiag);
        CHECK(s.op().name() == "ebe");
        REQUIRE(s.constrained_matrix() != nullptr);
        CHECK(s.constrained_matrix()->coeff(0, 0) == 1.0); // constrained row
        CHECK(s.constrained_matrix()->coeff(0, 1) == 0.0);
        CHECK(s.preconditioner().name() == "rcm+ic0");
        CHECK(s.description() == "PCG (CG + rcm+ic0, ebe)");
    }
}

TEST_CASE("PressureSolver builds pattern and ordering once per mesh", "[pressure_solver][configurations]") {
    const Mesh mesh = grid_mesh();
    PressureBC bc;
    bc.nodes = {0};
    bc.values = {0.0};
    PressureSolver s(configured("ic0", "rcm", "csr"), bc);
    std::vector<double> elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;

    update_lhs(mesh, step_dt(mesh.nelem, 0, true), elcoe2, pdiag, gstif);
    s.update(mesh, gstif, pdiag);
    const auto pattern = s.assembled_matrix()->pattern_ptr();
    const auto& reordered = dynamic_cast<const la::ReorderedPreconditioner&>(s.preconditioner());
    const la::Permutation* permutation = &reordered.permutation();
    const std::vector<int> order = permutation->new_to_old();

    for (int k = 1; k < 4; ++k) {
        update_lhs(mesh, step_dt(mesh.nelem, k, true), elcoe2, pdiag, gstif);
        s.update(mesh, gstif, pdiag);
        CHECK(s.assembled_matrix()->pattern_ptr() == pattern);
        CHECK(&reordered.permutation() == permutation);
        CHECK(reordered.permutation().new_to_old() == order);
    }

    SECTION("a different mesh triggers a rebuild") {
        GridSpec g;
        g.nx = 4;
        g.ny = 3;
        Mesh other = make_grid_mesh(g);
        {
            CaptureStdout quiet;
            compute_all_geometry(other);
        }
        update_lhs(other, std::vector<double>(other.nelem, 1e-2), elcoe2, pdiag, gstif);
        s.update(other, gstif, pdiag);
        CHECK(s.assembled_matrix()->rows() == other.npoin);
        std::vector<double> p(other.npoin, 0.0);
        CHECK(s.solve(test_vector(other.npoin), p).converged);
    }
}

TEST_CASE("CSR operator with legacy criteria tracks the legacy PCG", "[pressure_solver][configurations]") {
    // Same method, same criteria; only the summation order of the product
    // differs. At the loose bundled tolerance the iterate is far from
    // converged and round-off differences grow over ~100 CG iterations, so
    // the comparison is on the convergence history (iterations, residuals),
    // not on the unconverged iterates.
    const Mesh mesh = benchmark_mesh();
    PressureBC bc;
    {
        CaptureStdout quiet;
        bc = setup_pressure_bcs(mesh, 0);
    }
    SolverConfig csr = configured("jacobi", "none", "csr", 0.1, 1e-6);
    csr.residual_reference = "first_iteration";
    PressureSolver a(solver_config(0.1, 1e-6), bc), b(csr, bc);
    std::vector<double> elcoe2, pdiag, pa(mesh.npoin, 1e-4), pb(mesh.npoin, 1e-4);
    std::vector<std::array<double, 3>> gstif;
    update_lhs(mesh, step_dt(mesh.nelem, 0, false), elcoe2, pdiag, gstif);
    a.update(mesh, gstif, pdiag);
    b.update(mesh, gstif, pdiag);
    std::vector<double> rhs = test_vector(mesh.npoin);
    la::scale(rhs, 1e-4);
    const la::SolveResult ra = a.solve(rhs, pa), rb = b.solve(rhs, pb);
    CAPTURE(ra.iterations, rb.iterations, ra.final_residual, rb.final_residual);
    CHECK(std::abs(ra.iterations - rb.iterations) <= 1);
    CHECK(ra.reason == rb.reason);
    CHECK(std::abs(ra.reference_residual - rb.reference_residual) <= 1e-10 * ra.reference_residual);
    CHECK(std::abs(ra.final_residual - rb.final_residual) <= 1e-2 * ra.final_residual);
}

TEST_CASE("PressureSolver rejects invalid settings", "[pressure_solver][configurations]") {
    PressureBC bc;
    bc.nodes = {0};
    bc.values = {0.0};
    CHECK_THROWS_AS(PressureSolver(configured("amg", "none", "ebe"), bc), std::invalid_argument);
    CHECK_THROWS_AS(PressureSolver(configured("ic0", "none", "coo"), bc), std::invalid_argument);
    CHECK_THROWS_AS(PressureSolver(configured("jacobi", "rcm", "ebe"), bc), std::invalid_argument);
    SolverConfig c = configured("ic0", "none", "ebe");
    c.method = "minres";
    CHECK_THROWS_AS(PressureSolver(c, bc), std::invalid_argument);
    c.method = "gmres";
    c.gmres_orthogonalization = "householder";
    CHECK_THROWS_AS(PressureSolver(c, bc), std::invalid_argument);
    c.gmres_orthogonalization = "mgs";
    c.gmres_restart = 0;
    CHECK_THROWS_AS(PressureSolver(c, bc), std::invalid_argument);
}

TEST_CASE("PressureSolver with non-symmetric Krylov methods", "[pressure_solver][configurations]") {
    PressureBC bc;
    bc.nodes = {0};
    bc.values = {0.0};
    SolverConfig g = configured("ilu0", "rcm", "csr", 1e-8, 0.0, "gmres");
    g.gmres_restart = 12;
    g.gmres_orthogonalization = "cgs2";
    const PressureSolver gmres(g, bc);
    CHECK(gmres.description() == "Krylov (GMRES(12) + rcm+ilu0, csr)");
    CHECK(gmres.label() == "GMRES(12)");
    CHECK_FALSE(gmres.legacy());
    CHECK(gmres.krylov().criteria().reference == la::ResidualReference::InitialResidual);
    const auto& impl = dynamic_cast<const la::Gmres&>(gmres.krylov());
    CHECK(impl.options().orthogonalization == la::Orthogonalization::ClassicalGramSchmidtTwice);

    const PressureSolver bicgstab(configured("jacobi", "none", "ebe", 1e-8, 0.0, "bicgstab"), bc);
    CHECK(bicgstab.description() == "Krylov (BiCGStab + jacobi, ebe)");
    CHECK(bicgstab.label() == "BiCGStab");
    CHECK(PressureSolver(solver_config(0.1, 1e-6), bc).label() == "PCG");
}
