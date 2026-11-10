// Optional linear solver settings of *.solver (IO::read_solver): the
// "key = value" block after the positional values, its validation, and the
// derived convergence criteria (pressure_criteria).
//
// Syntax: after the 10 positional lines, a line is a setting iff it starts
// with an identifier followed by '='; '#' or '!' start a comment; other
// lines (headers, separators) are ignored. Keys are case-insensitive, so are
// the names of methods/preconditioners. Unknown keys, invalid values and
// duplicate keys make read_solver fail.

#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "PressureSolver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace hydrama;
using namespace hydrama::test;

namespace {

const std::string kPositional =
    "-----\n"
    "Pressure Field Solver (1 - Conjugate Gradient)\n"
    "1\n"
    "-----\n"
    "Conjugate Gradient Convergence Tolerances\n"
    "Relative Residual. Zero for OFF\n"
    "0.01\n"
    "-----\n"
    "Absolute Residual. Zero for OFF\n"
    "1.0d-08\n";

struct ReadResult {
    bool ok = false;
    SolverConfig config;
    std::string errors;
    std::string output;
};

ReadResult read_with(const std::string& options) {
    TempDir dir("solver-options");
    const fs::path file = dir / "case.solver";
    write_text(file, kPositional + options);
    ReadResult r;
    CaptureStderr err;
    CaptureStdout out;
    r.ok = IO::read_solver(file.string(), r.config);
    r.errors = err.str();
    r.output = out.str();
    return r;
}

} // namespace

TEST_CASE("Without options the solver file selects the legacy PCG", "[io][config][solver_options]") {
    const ReadResult r = read_with("");
    REQUIRE(r.ok);
    CHECK(r.config.relToler == 0.01);
    CHECK(r.config.absToler == 1e-8);
    CHECK(r.config.method == "cg");
    CHECK(r.config.preconditioner == "jacobi");
    CHECK(r.config.operator_format == "ebe");
    CHECK(r.config.reordering == "none");
    CHECK(r.config.max_iterations == 100000);
    CHECK(r.config.residual_reference.empty());
    CHECK(r.config.ssor_omega == 1.0);
    CHECK(r.config.diagonal_shift == 0.0);
    CHECK(r.config.is_legacy_pcg());
}

TEST_CASE("The bundled solver file documents the defaults", "[io][config][solver_options]") {
    SolverConfig c;
    CaptureStdout quiet;
    REQUIRE(IO::read_solver(std::string(HYDRAMA2D_CASE_DIR) + "/system/hinterkante.solver", c));
    CHECK(c.is_legacy_pcg());
    CHECK(c.max_iterations == 100000);
    CHECK(c.ssor_omega == 1.0);
    CHECK(c.diagonal_shift == 0.0);
    CHECK(c.residual_reference.empty());
    CHECK(c.method == "cg");
    CHECK(c.gmres_restart == 30);
    CHECK(c.gmres_orthogonalization == "mgs");
    CHECK(c.preconditioner_reuse == "rescale");
}

TEST_CASE("All options are read", "[io][config][solver_options]") {
    const ReadResult r = read_with(
        "-----\n"
        "Linear Solver Options (key = value)\n"
        "method = CG\n"
        "  Preconditioner=IC0   # comment\n"
        "operator_format = csr ! Fortran-style comment\n"
        "reordering = rcm\n"
        "max_iterations = 2500\n"
        "residual_reference = rhs\n"
        "ssor_omega = 1.2d0\n"
        "diagonal_shift = 1.0e-2\n");
    INFO(r.errors);
    REQUIRE(r.ok);
    CHECK(r.config.method == "cg");
    CHECK(r.config.preconditioner == "ic0");
    CHECK(r.config.operator_format == "csr");
    CHECK(r.config.reordering == "rcm");
    CHECK(r.config.max_iterations == 2500);
    CHECK(r.config.residual_reference == "rhs");
    CHECK(r.config.ssor_omega == 1.2);
    CHECK(r.config.diagonal_shift == 1e-2);
    CHECK_FALSE(r.config.is_legacy_pcg());
    CHECK(r.errors.empty());
}

TEST_CASE("Non-symmetric Krylov methods and their options are read", "[io][config][solver_options]") {
    const ReadResult r = read_with("method = GMRES\n"
                                   "preconditioner = ilu0\n"
                                   "gmres_restart = 45\n"
                                   "gmres_orthogonalization = CGS2\n");
    INFO(r.errors);
    REQUIRE(r.ok);
    CHECK(r.config.method == "gmres");
    CHECK(r.config.gmres_restart == 45);
    CHECK(r.config.gmres_orthogonalization == "cgs2");
    CHECK_FALSE(r.config.is_legacy_pcg());
    CHECK(r.output.find("WARNING") == std::string::npos); // ilu0 fits gmres

    const ReadResult b = read_with("method = bicgstab\n");
    REQUIRE(b.ok);
    CHECK(b.config.method == "bicgstab");
    CHECK(b.config.preconditioner == "jacobi");
    CHECK(b.config.gmres_restart == 30); // default
    CHECK(b.config.gmres_orthogonalization == "mgs");
}

TEST_CASE("Free text and separators between options are ignored", "[io][config][solver_options]") {
    const ReadResult r = read_with(
        "=====================\n"
        "Choose: preconditioner = ssor gives SGS for omega = 1\n" // not an identifier before '='
        "# preconditioner = ic0\n"                                  // comment
        "preconditioner = ssor\n");
    INFO(r.errors);
    REQUIRE(r.ok);
    CHECK(r.config.preconditioner == "ssor");
}

TEST_CASE("Invalid options make read_solver fail with a message", "[io][config][solver_options]") {
    const std::string options = GENERATE(std::string("precondtioner = ic0\n"),              // unknown key
                                         std::string("preconditioner = amg\n"),             // unknown name
                                         std::string("method = minres\n"),                  // not available
                                         std::string("gmres_restart = 0\n"),
                                         std::string("gmres_restart = many\n"),
                                         std::string("gmres_orthogonalization = householder\n"),
                                         std::string("operator_format = coo\n"),
                                         std::string("reordering = amd\n"),
                                         std::string("max_iterations = 0\n"),
                                         std::string("max_iterations = 10.5\n"),
                                         std::string("residual_reference = final\n"),
                                         std::string("ssor_omega = 2.0\n"),
                                         std::string("ssor_omega = 0\n"),
                                         std::string("ssor_omega = fast\n"),
                                         std::string("diagonal_shift = -0.1\n"),
                                         std::string("preconditioner =\n"),                 // missing value
                                         std::string("preconditioner = ic0\npreconditioner = ssor\n"),  // duplicate
                                         std::string("reordering = rcm\n"),                 // with jacobi
                                         std::string("preconditioner = none\nreordering = rcm\n"));
    CAPTURE(options);
    const ReadResult r = read_with(options);
    CHECK_FALSE(r.ok);
    CHECK(r.errors.find("ERROR: *.solver OPTION") != std::string::npos);
}

TEST_CASE("Reordering is accepted for every assembled preconditioner", "[io][config][solver_options]") {
    const std::string name = GENERATE(std::string("ssor"), std::string("ic0"), std::string("ilu0"));
    const ReadResult r = read_with("preconditioner = " + name + "\nreordering = rcm\n");
    INFO(r.errors);
    CHECK(r.ok);
}

TEST_CASE("A non-symmetric preconditioner with CG gives a warning", "[io][config][solver_options]") {
    const ReadResult r = read_with("preconditioner = ilu0\n");
    REQUIRE(r.ok);
    CHECK(r.output.find("WARNING") != std::string::npos);
    CHECK(read_with("preconditioner = ic0\n").output.find("WARNING") == std::string::npos);
}

TEST_CASE("Positional values keep their original parsing", "[io][config][solver_options]") {
    // Options do not change how the first ten lines are read.
    const ReadResult r = read_with("preconditioner = ic0\n");
    REQUIRE(r.ok);
    CHECK(r.config.solver_opt == 1);
    CHECK(r.config.relToler == 0.01);
    CHECK(r.config.absToler == 1e-8);
}

// ---------------------------------------------------------------------------
// Derived convergence criteria
// ---------------------------------------------------------------------------

TEST_CASE("Convergence criteria derived from the settings", "[io][config][solver_options]") {
    SolverConfig c;
    c.relToler = 0.05;
    c.absToler = 1e-7;

    SECTION("legacy PCG: first-iteration reference, continue on breakdown") {
        const linalg::ConvergenceCriteria k = pressure_criteria(c);
        CHECK(k.reference == linalg::ResidualReference::FirstIteration);
        CHECK(k.on_breakdown == linalg::BreakdownPolicy::Continue);
        CHECK(k.rel_tol == 0.05);
        CHECK(k.abs_tol == 1e-7);
        CHECK(k.max_iterations == 100000);
    }
    SECTION("any other configuration: initial-residual reference, stop on breakdown") {
        const int which = GENERATE(0, 1, 2);
        if (which == 0) c.preconditioner = "ic0";
        if (which == 1) c.operator_format = "csr";
        if (which == 2) c.preconditioner = "none";
        const linalg::ConvergenceCriteria k = pressure_criteria(c);
        CHECK(k.reference == linalg::ResidualReference::InitialResidual);
        CHECK(k.on_breakdown == linalg::BreakdownPolicy::Stop);
        CHECK(k.rel_tol == 0.05);
    }
    SECTION("an explicit residual_reference and max_iterations are honoured") {
        c.preconditioner = "ssor";
        c.residual_reference = "first_iteration";
        c.max_iterations = 321;
        linalg::ConvergenceCriteria k = pressure_criteria(c);
        CHECK(k.reference == linalg::ResidualReference::FirstIteration);
        CHECK(k.max_iterations == 321);
        c.residual_reference = "rhs";
        CHECK(pressure_criteria(c).reference == linalg::ResidualReference::RightHandSide);
        c.residual_reference = "bogus";
        CHECK_THROWS_AS(pressure_criteria(c), std::invalid_argument);
    }
}
