// linalg::Span views and the vector kernels of linalg/VectorOps.hpp:
// values, size checks, permitted aliasing and the deterministic (sequential,
// index-ordered) reductions the solvers rely on for bit-for-bit
// reproducibility.

#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <type_traits>

using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Vector views reference the underlying storage", "[linalg][vector]") {
    std::vector<double> v = {1.0, 2.0, 3.0};
    la::VectorView view = v;
    REQUIRE(view.size() == 3);
    CHECK(view.data() == v.data());
    CHECK_FALSE(view.empty());

    view[1] = 20.0;
    CHECK(v[1] == 20.0);

    double sum = 0.0;
    for (double x : view) sum += x;
    CHECK(sum == 24.0);

    SECTION("mutable views convert to read-only views") {
        la::ConstVectorView cview = view;
        CHECK(cview.data() == v.data());
        CHECK(cview.size() == 3);
    }
    SECTION("read-only views bind to const vectors") {
        const std::vector<double>& cv = v;
        la::ConstVectorView cview = cv;
        CHECK(cview[2] == 3.0);
    }
    SECTION("views of raw memory and empty views") {
        la::VectorView part(v.data() + 1, 2);
        CHECK(part[0] == 20.0);
        CHECK(part.size() == 2);
        la::ConstVectorView none;
        CHECK(none.empty());
        CHECK(none.begin() == none.end());
    }
}

TEST_CASE("Vector view conversions are const-correct", "[linalg][vector]") {
    STATIC_REQUIRE(std::is_convertible_v<std::vector<double>&, la::VectorView>);
    STATIC_REQUIRE(std::is_convertible_v<std::vector<double>&, la::ConstVectorView>);
    STATIC_REQUIRE(std::is_convertible_v<const std::vector<double>&, la::ConstVectorView>);
    STATIC_REQUIRE(std::is_convertible_v<la::VectorView, la::ConstVectorView>);
    STATIC_REQUIRE_FALSE(std::is_convertible_v<const std::vector<double>&, la::VectorView>);
    STATIC_REQUIRE_FALSE(std::is_convertible_v<la::ConstVectorView, la::VectorView>);
}

TEST_CASE("Vector kernels compute the expected values", "[linalg][vector]") {
    const std::vector<double> x = {1.0, -2.0, 3.0, 0.5};
    const std::vector<double> y = {4.0, 1.0, -1.0, 2.0};

    CHECK(la::dot(x, y) == 4.0 - 2.0 - 3.0 + 1.0);
    CHECK_THAT(la::norm2(x), WithinRel(std::sqrt(1.0 + 4.0 + 9.0 + 0.25), 1e-15));

    std::vector<double> z(4);
    la::copy(x, z);
    CHECK(z == x);

    la::fill(z, 7.0);
    CHECK(z == std::vector<double>(4, 7.0));

    z = x;
    la::scale(z, -2.0);
    CHECK(z == std::vector<double>{-2.0, 4.0, -6.0, -1.0});

    z = y;
    la::axpy(2.0, x, z); // z = y + 2x
    CHECK(z == std::vector<double>{6.0, -3.0, 5.0, 3.0});

    z = y;
    la::xpby(x, 0.5, z); // z = x + 0.5 y
    CHECK(z == std::vector<double>{3.0, -1.5, 2.5, 1.5});

    la::subtract(x, y, z);
    CHECK(z == std::vector<double>{-3.0, -3.0, 4.0, -1.5});

    la::pointwise_multiply(x, y, z);
    CHECK(z == std::vector<double>{4.0, -2.0, -3.0, 1.0});
}

TEST_CASE("Vector kernels allow documented aliasing", "[linalg][vector]") {
    std::vector<double> x = {1.0, 2.0, 3.0};
    const std::vector<double> y = {0.5, 0.5, 2.0};
    la::subtract(x, y, x);
    CHECK(x == std::vector<double>{0.5, 1.5, 1.0});
    la::pointwise_multiply(x, y, x);
    CHECK(x == std::vector<double>{0.25, 0.75, 2.0});
    // dot of a vector with itself is its squared norm
    CHECK_THAT(la::dot(x, x), WithinRel(la::norm2(x) * la::norm2(x), 1e-15));
}

TEST_CASE("Vector kernels reject size mismatches", "[linalg][vector]") {
    std::vector<double> a(3, 1.0), b(4, 1.0);
    CHECK_THROWS_AS(la::dot(a, b), std::invalid_argument);
    CHECK_THROWS_AS(la::copy(a, b), std::invalid_argument);
    CHECK_THROWS_AS(la::axpy(1.0, a, b), std::invalid_argument);
    CHECK_THROWS_AS(la::xpby(a, 1.0, b), std::invalid_argument);
    CHECK_THROWS_AS(la::subtract(a, a, b), std::invalid_argument);
    CHECK_THROWS_AS(la::pointwise_multiply(a, b, a), std::invalid_argument);
}

TEST_CASE("Vector kernels handle empty vectors", "[linalg][vector]") {
    std::vector<double> e;
    CHECK(la::dot(e, e) == 0.0);
    CHECK(la::norm2(e) == 0.0);
    CHECK_NOTHROW(la::axpy(2.0, e, e));
    CHECK_NOTHROW(la::fill(e, 1.0));
}

TEST_CASE("Reductions accumulate sequentially in index order", "[linalg][vector]") {
    // Bit-for-bit reproducibility of the solvers relies on the reduction
    // order: the result must equal a plain left-to-right loop exactly, for
    // data whose sum depends on the order (mixed magnitudes).
    const int n = 10007;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = std::sin(0.37 * i) * std::pow(10.0, (i % 17) - 8);
        y[i] = std::cos(1.3 * i) + 1e-9 * i;
    }
    double d = 0.0, s = 0.0;
    for (int i = 0; i < n; ++i) {
        d += x[i] * y[i];
        s += x[i] * x[i];
    }
    CHECK(la::dot(x, y) == d);
    CHECK(la::norm2(x) == std::sqrt(s));

    // Sanity check that the data is order sensitive (otherwise the test
    // above would be vacuous).
    double reversed = 0.0;
    for (int i = n - 1; i >= 0; --i) reversed += x[i] * y[i];
    CHECK(reversed != d);
}
