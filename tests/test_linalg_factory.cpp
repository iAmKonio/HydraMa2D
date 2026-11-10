// Name-based creation of preconditioners and Krylov solvers (the catalogues
// the configuration reader validates against).

#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace hydrama::test;

TEST_CASE("Preconditioner catalogue", "[linalg][factory]") {
    const auto& cat = la::preconditioner_catalog();
    std::vector<std::string_view> names;
    for (const auto& info : cat) names.push_back(info.name);
    CHECK(names == std::vector<std::string_view>{"none", "jacobi", "ssor", "ic0", "ilu0"});
    CHECK(la::preconditioner_names() == "none | jacobi | ssor | ic0 | ilu0");

    CHECK_FALSE(la::find_preconditioner("none")->requires_assembled);
    CHECK_FALSE(la::find_preconditioner("jacobi")->requires_assembled);
    CHECK(la::find_preconditioner("ssor")->requires_assembled);
    CHECK(la::find_preconditioner("ic0")->requires_assembled);
    CHECK(la::find_preconditioner("ilu0")->requires_assembled);
    CHECK(la::find_preconditioner("ic0")->symmetric);
    CHECK_FALSE(la::find_preconditioner("ilu0")->symmetric);
    CHECK(la::find_preconditioner("amg") == nullptr);
    for (const auto& info : cat) CHECK_FALSE(info.description.empty());
}

TEST_CASE("make_preconditioner creates every catalogued preconditioner", "[linalg][factory]") {
    const la::CsrMatrix a = from_dense(laplacian_1d(8), true);
    for (const auto& info : la::preconditioner_catalog()) {
        CAPTURE(info.name);
        auto m = la::make_preconditioner(info.name);
        REQUIRE(m);
        CHECK(m->name() == info.name);
        m->setup(a);
        CHECK(m->size() == 8);
        if (info.symmetric) CHECK(m->traits().symmetric); // for a symmetric matrix
    }
}

TEST_CASE("make_preconditioner passes options and wraps orderings", "[linalg][factory]") {
    la::PreconditionerOptions o;
    o.ssor_omega = 1.3;
    auto ssor = la::make_preconditioner("ssor", o);
    CHECK(dynamic_cast<la::SsorPreconditioner&>(*ssor).omega() == 1.3);

    o.diagonal_shift = 0.5;
    auto ic = la::make_preconditioner("ic0", o);
    ic->setup(from_dense(laplacian_1d(4), true));
    CHECK(dynamic_cast<la::IncompleteCholesky&>(*ic).shift() == 0.5);

    o.ordering = "rcm";
    const std::string inner = GENERATE(std::string("ssor"), std::string("ic0"), std::string("ilu0"));
    auto reordered = la::make_preconditioner(inner, o);
    CHECK(reordered->name() == "rcm+" + inner);
    CHECK(dynamic_cast<la::ReorderedPreconditioner*>(reordered.get()) != nullptr);
}

TEST_CASE("make_preconditioner rejects invalid requests", "[linalg][factory]") {
    CHECK_THROWS_AS(la::make_preconditioner("amg"), std::invalid_argument);
    la::PreconditionerOptions o;
    o.ordering = "rcm";
    CHECK_THROWS_AS(la::make_preconditioner("jacobi", o), std::invalid_argument); // matrix-free
    CHECK_THROWS_AS(la::make_preconditioner("none", o), std::invalid_argument);
    o.ordering = "amd";
    CHECK_THROWS_AS(la::make_preconditioner("ic0", o), std::invalid_argument);
    la::PreconditionerOptions bad_omega;
    bad_omega.ssor_omega = 2.0;
    CHECK_THROWS_AS(la::make_preconditioner("ssor", bad_omega), std::invalid_argument);
    la::PreconditionerOptions bad_shift;
    bad_shift.diagonal_shift = -1.0;
    CHECK_THROWS_AS(la::make_preconditioner("ilu0", bad_shift), std::invalid_argument);
}

TEST_CASE("Krylov solver catalogue and creation", "[linalg][factory]") {
    std::vector<std::string_view> names;
    for (const auto& info : la::krylov_catalog()) names.push_back(info.name);
    CHECK(names == std::vector<std::string_view>{"cg", "bicgstab", "gmres"});
    CHECK(la::krylov_solver_names() == "cg | bicgstab | gmres");
    CHECK(la::find_krylov_solver("cg")->requires_symmetric);
    CHECK_FALSE(la::find_krylov_solver("bicgstab")->requires_symmetric);
    CHECK_FALSE(la::find_krylov_solver("gmres")->requires_symmetric);
    CHECK(la::find_krylov_solver("minres") == nullptr);

    la::ConvergenceCriteria c;
    c.rel_tol = 1e-5;
    c.max_iterations = 77;
    for (const auto& [name, expected] : std::vector<std::pair<std::string, std::string>>{
             {"cg", "CG"}, {"bicgstab", "BiCGStab"}, {"gmres", "GMRES(30)"}}) {
        CAPTURE(name);
        auto solver = la::make_krylov_solver(name, c);
        CHECK(solver->name() == expected);
        CHECK(solver->criteria().rel_tol == 1e-5);
        CHECK(solver->criteria().max_iterations == 77);
    }
    CHECK_THROWS_AS(la::make_krylov_solver("minres"), std::invalid_argument);
}

TEST_CASE("make_krylov_solver passes the GMRES options", "[linalg][factory]") {
    la::KrylovOptions o;
    o.gmres_restart = 12;
    o.gmres_orthogonalization = "cgs2";
    auto solver = la::make_krylov_solver("gmres", {}, o);
    auto& gmres = dynamic_cast<la::Gmres&>(*solver);
    CHECK(gmres.name() == "GMRES(12)");
    CHECK(gmres.options().restart == 12);
    CHECK(gmres.options().orthogonalization == la::Orthogonalization::ClassicalGramSchmidtTwice);

    CHECK(la::is_orthogonalization("mgs"));
    CHECK(la::is_orthogonalization("cgs2"));
    CHECK_FALSE(la::is_orthogonalization("householder"));
    o.gmres_orthogonalization = "householder";
    CHECK_THROWS_AS(la::make_krylov_solver("gmres", {}, o), std::invalid_argument);
    o.gmres_orthogonalization = "mgs";
    o.gmres_restart = 0;
    CHECK_THROWS_AS(la::make_krylov_solver("gmres", {}, o), std::invalid_argument);
    // Options irrelevant to a method are ignored.
    CHECK_NOTHROW(la::make_krylov_solver("bicgstab", {}, o));
}
