#include "MeshTestUtils.hpp"
#include "QuadTestUtils.hpp"
#include "MeshCaseTestUtils.hpp"
#include "IO.hpp"
#include "Step1.hpp"
#include "Step2.hpp"
#include "Step3.hpp"
#include "LinearSolver.hpp"
#include "linalg/CsrMatrix.hpp"
#include "linalg/EbeOperator.hpp"
#include "linalg/VectorOps.hpp"
#include "numerics/OpenMP.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <set>
#include <vector>
#include <iomanip>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("OpenMP thread control and queries", "[openmp]") {
    const int initial_threads = get_max_threads();
    CHECK(initial_threads >= 1);
    CHECK(get_thread_num() == 0);
    CHECK(!in_parallel());

    set_num_threads(2);
#ifdef HYDRAMA2D_OPENMP
    CHECK(get_max_threads() == 2);
#else
    CHECK(get_max_threads() == 1);
#endif

    set_num_threads(initial_threads);
}

TEST_CASE("Mesh element coloring correctness on triangles and quads", "[openmp][mesh]") {
    SECTION("Pure triangle grid") {
        Mesh m = make_grid_mesh(GridSpec{0.0, 0.0, 1.0, 1.0, 6, 6});
        compute_geometry(m);

        REQUIRE(!m.triangle_color_groups.empty());
        int total_colored = 0;
        std::vector<int> count_per_elem(m.num_triangles(), 0);

        for (const auto& grp : m.triangle_color_groups) {
            std::set<int> touched_nodes;
            for (int ie : grp) {
                REQUIRE(ie >= 0);
                REQUIRE(ie < m.num_triangles());
                count_per_elem[ie]++;
                total_colored++;
                for (int in = 0; in < 3; ++in) {
                    int ip = m.intma[ie][in];
                    // No two elements in the same color group share any node
                    auto [it, inserted] = touched_nodes.insert(ip);
                    REQUIRE(inserted);
                }
            }
        }
        CHECK(total_colored == m.num_triangles());
        for (int c : count_per_elem) {
            CHECK(c == 1);
        }
    }

    SECTION("Pure quad grid") {
        GridOptions opts;
        opts.nx = 6;
        opts.ny = 6;
        opts.cells = GridCells::Quads;
        Mesh m = make_hybrid_mesh(grid(opts));
        m.compute_shape_function_derivatives();

        REQUIRE(!m.quad_color_groups.empty());
        int total_colored = 0;
        std::vector<int> count_per_elem(m.num_quads(), 0);

        for (const auto& grp : m.quad_color_groups) {
            std::set<int> touched_nodes;
            for (int q : grp) {
                REQUIRE(q >= 0);
                REQUIRE(q < m.num_quads());
                count_per_elem[q]++;
                total_colored++;
                for (int in = 0; in < 4; ++in) {
                    int ip = m.quads[q][in];
                    auto [it, inserted] = touched_nodes.insert(ip);
                    REQUIRE(inserted);
                }
            }
        }
        CHECK(total_colored == m.num_quads());
        for (int c : count_per_elem) {
            CHECK(c == 1);
        }
    }

    SECTION("Hybrid Gmsh mesh cylinder_inflation_4.1_binary.msh") {
        Mesh m;
        std::string path = std::string(HYDRAMA2D_TEST_DATA_DIR) + "/gmsh/cylinder_inflation_4.1_binary.msh";
        REQUIRE(IO::read_gmsh_mesh(path, "", m));
        m.compute_shape_function_derivatives();

        REQUIRE(!m.triangle_color_groups.empty());
        for (const auto& grp : m.triangle_color_groups) {
            std::set<int> touched_nodes;
            for (int ie : grp) {
                for (int in = 0; in < 3; ++in) {
                    int ip = m.intma[ie][in];
                    auto [it, inserted] = touched_nodes.insert(ip);
                    CHECK(inserted);
                }
            }
        }

        REQUIRE(!m.quad_color_groups.empty());
        for (const auto& grp : m.quad_color_groups) {
            std::set<int> touched_nodes;
            for (int q : grp) {
                for (int in = 0; in < 4; ++in) {
                    int ip = m.quads[q][in];
                    auto [it, inserted] = touched_nodes.insert(ip);
                    CHECK(inserted);
                }
            }
        }

        const int npoin = m.npoin;
        const int nelem = m.nelem;
        std::vector<double> u(npoin), v(npoin), u_star(npoin), v_star(npoin);
        std::vector<double> p_new(npoin), p_old(npoin);
        std::vector<double> delte(nelem, 0.01);
        for (int i = 0; i < npoin; ++i) {
            u[i] = 1.0 + 0.1 * m.x[i] + 0.2 * m.y[i];
            v[i] = 0.5 - 0.2 * m.x[i] + 0.1 * m.y[i];
            u_star[i] = u[i] + 0.01;
            v_star[i] = v[i] - 0.01;
            p_new[i] = 100.0 + 10.0 * m.x[i];
            p_old[i] = 99.0 + 9.5 * m.x[i];
        }
        const double ani = 0.01;
        const double theta1 = 1.0;

        int orig_threads = get_max_threads();

        set_num_threads(1);
        std::vector<double> rhs_u1_s1, rhs_v1_s1, rhs1_s2, rhs_u1_s3, rhs_v1_s3;
        compute_step1(m, ani, delte, u, v, rhs_u1_s1, rhs_v1_s1);
        compute_step2(m, u_star, v_star, u, v, theta1, rhs1_s2);
        compute_step3(m, u, v, p_new, p_old, delte, rhs_u1_s3, rhs_v1_s3);

        set_num_threads(4);
        std::vector<double> rhs_u4_s1, rhs_v4_s1, rhs4_s2, rhs_u4_s3, rhs_v4_s3;
        compute_step1(m, ani, delte, u, v, rhs_u4_s1, rhs_v4_s1);
        compute_step2(m, u_star, v_star, u, v, theta1, rhs4_s2);
        compute_step3(m, u, v, p_new, p_old, delte, rhs_u4_s3, rhs_v4_s3);

        CHECK(rhs_u1_s1 == rhs_u4_s1);
        CHECK(rhs_v1_s1 == rhs_v4_s1);
        CHECK(rhs1_s2 == rhs4_s2);
        CHECK(rhs_u1_s3 == rhs_u4_s3);
        CHECK(rhs_v1_s3 == rhs_v4_s3);

        set_num_threads(orig_threads);
    }

    SECTION("Hybrid mesh full case bit reproducibility across thread counts") {
        std::string msh_content = data_file("cylinder_inflation_4.1_binary.msh");
        CaseOptions opts;
        opts.steps = 5;

        int orig_threads = get_max_threads();

        set_num_threads(1);
        const MeshCaseResult r1 = run_mesh_case("hybrid-threads-1", msh_content, opts);

        set_num_threads(4);
        const MeshCaseResult r4 = run_mesh_case("hybrid-threads-4", msh_content, opts);

        CHECK(r1.run.u == r4.run.u);
        CHECK(r1.run.v == r4.run.v);
        CHECK(r1.run.p == r4.run.p);

        set_num_threads(orig_threads);
    }
}

TEST_CASE("CBS Steps multi-threading bit reproducibility", "[openmp][step1][step2][step3]") {
    Mesh m = make_grid_mesh(GridSpec{0.0, 0.0, 1.0, 1.0, 8, 8});
    compute_geometry(m);

    const int npoin = m.npoin;
    const int nelem = m.nelem;
    std::vector<double> u(npoin), v(npoin), u_star(npoin), v_star(npoin);
    std::vector<double> p_new(npoin), p_old(npoin);
    std::vector<double> delte(nelem, 0.01);
    for (int i = 0; i < npoin; ++i) {
        u[i] = 1.0 + 0.1 * m.x[i] + 0.2 * m.y[i];
        v[i] = 0.5 - 0.2 * m.x[i] + 0.1 * m.y[i];
        u_star[i] = u[i] + 0.01;
        v_star[i] = v[i] - 0.01;
        p_new[i] = 100.0 + 10.0 * m.x[i];
        p_old[i] = 99.0 + 9.5 * m.x[i];
    }
    const double ani = 0.01;
    const double theta1 = 1.0;

    int orig_threads = get_max_threads();

    // 1-thread execution
    set_num_threads(1);
    std::vector<double> rhs_u1_s1, rhs_v1_s1;
    compute_step1(m, ani, delte, u, v, rhs_u1_s1, rhs_v1_s1);

    std::vector<double> rhs1_s2;
    compute_step2(m, u_star, v_star, u, v, theta1, rhs1_s2);

    std::vector<double> rhs_u1_s3, rhs_v1_s3;
    compute_step3(m, u, v, p_new, p_old, delte, rhs_u1_s3, rhs_v1_s3);

    // Multi-threaded execution (e.g. 4 threads)
    set_num_threads(4);
    std::vector<double> rhs_u4_s1, rhs_v4_s1;
    compute_step1(m, ani, delte, u, v, rhs_u4_s1, rhs_v4_s1);

    std::vector<double> rhs4_s2;
    compute_step2(m, u_star, v_star, u, v, theta1, rhs4_s2);

    std::vector<double> rhs_u4_s3, rhs_v4_s3;
    compute_step3(m, u, v, p_new, p_old, delte, rhs_u4_s3, rhs_v4_s3);

    // Bit-identical comparisons!
    CHECK(rhs_u1_s1 == rhs_u4_s1);
    CHECK(rhs_v1_s1 == rhs_v4_s1);

    CHECK(rhs1_s2 == rhs4_s2);

    CHECK(rhs_u1_s3 == rhs_u4_s3);
    CHECK(rhs_v1_s3 == rhs_v4_s3);

    set_num_threads(orig_threads);
}

TEST_CASE("Linear assembly and operators thread reproducibility", "[openmp][linalg]") {
    Mesh m = make_grid_mesh(GridSpec{0.0, 0.0, 1.0, 1.0, 8, 8});
    compute_geometry(m);
    m.compute_laplacian_stiffness();

    const int npoin = m.npoin;
    const int nelem = m.nelem;
    std::vector<double> delte(nelem, 0.005);
    std::vector<double> x(npoin, 1.23);

    int orig_threads = get_max_threads();

    // 1-thread
    set_num_threads(1);
    std::vector<double> elcoe2_1, pdiag_1;
    std::vector<std::array<double, 3>> gstif_1;
    update_lhs(m, delte, elcoe2_1, pdiag_1, gstif_1);

    std::vector<double> y_ebe_1(npoin);
    linalg::ebe_apply(m, gstif_1.data(), pdiag_1.data(), x.data(), y_ebe_1.data());

    // 4-threads
    set_num_threads(4);
    std::vector<double> elcoe2_4, pdiag_4;
    std::vector<std::array<double, 3>> gstif_4;
    update_lhs(m, delte, elcoe2_4, pdiag_4, gstif_4);

    std::vector<double> y_ebe_4(npoin);
    linalg::ebe_apply(m, gstif_4.data(), pdiag_4.data(), x.data(), y_ebe_4.data());

    CHECK(elcoe2_1 == elcoe2_4);
    CHECK(pdiag_1 == pdiag_4);
    CHECK(gstif_1 == gstif_4);
    CHECK(y_ebe_1 == y_ebe_4);

    // CSR Matrix apply
    std::vector<linalg::Triplet> triplets;
    for (int ie = 0; ie < m.num_triangles(); ++ie) {
        int n0 = m.intma[ie][0], n1 = m.intma[ie][1], n2 = m.intma[ie][2];
        triplets.push_back({n0, n1, gstif_1[ie][0]});
        triplets.push_back({n1, n0, gstif_1[ie][0]});
        triplets.push_back({n0, n2, gstif_1[ie][1]});
        triplets.push_back({n2, n0, gstif_1[ie][1]});
        triplets.push_back({n1, n2, gstif_1[ie][2]});
        triplets.push_back({n2, n1, gstif_1[ie][2]});
    }
    for (int i = 0; i < npoin; ++i) {
        triplets.push_back({i, i, pdiag_1[i]});
    }
    linalg::CsrMatrix csr = linalg::CsrMatrix::from_triplets(npoin, npoin, triplets);

    std::vector<double> y_csr_1(npoin, 0.0), y_csr_4(npoin, 0.0);
    set_num_threads(1);
    csr.apply(x, y_csr_1);

    set_num_threads(4);
    csr.apply(x, y_csr_4);

    CHECK(y_csr_1 == y_csr_4);

    // VectorOps
    std::vector<double> a(2000, 2.0), b(2000, 3.0), c1(2000, 0.0), c4(2000, 0.0);
    set_num_threads(1);
    linalg::axpy(1.5, a, c1);
    double dot1 = linalg::dot(a, b);
    double norm1 = linalg::norm2(a);

    set_num_threads(4);
    linalg::axpy(1.5, a, c4);
    double dot4 = linalg::dot(a, b);
    double norm4 = linalg::norm2(a);

    CHECK(c1 == c4);
    CHECK_THAT(dot1, WithinRel(dot4, 1e-14));
    CHECK_THAT(norm1, WithinRel(norm4, 1e-14));

    set_num_threads(orig_threads);
}
