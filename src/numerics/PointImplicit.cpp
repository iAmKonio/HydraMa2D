#include "numerics/PointImplicit.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace hydrama {

namespace {

template <bool kQuadratic>
void update(const std::vector<double>& nodal_mass, const std::vector<double>& elcoe2, const std::vector<double>& rhs,
            const std::vector<double>& production, const std::vector<double>& destruction,
            const std::vector<double>* quadratic, bool enforce_positivity, std::vector<double>& phi) {
    const std::size_t n = phi.size();
    if (nodal_mass.size() != n || elcoe2.size() != n || rhs.size() != n || production.size() != n ||
        destruction.size() != n || (kQuadratic && quadratic->size() != n)) {
        throw std::invalid_argument("point_implicit_update: size mismatch");
    }

    for (std::size_t i = 0; i < n; ++i) {
        const double dt = nodal_mass[i] * elcoe2[i];
        const double old = phi[i];
        const double explicit_increment = elcoe2[i] * rhs[i] + dt * production[i];
        double numerator = old;
        double denominator = 1.0 + dt * destruction[i];
        if (!enforce_positivity || explicit_increment >= 0.0) {
            numerator += explicit_increment;
        } else if (old > 0.0) {
            denominator -= explicit_increment / old;
        }
        if constexpr (kQuadratic) {
            const double b = (*quadratic)[i];
            if (b > 0.0) {
                // dt b phi^2 + denominator phi - numerator = 0, non-negative root.
                const double a2 = dt * b;
                if (!(numerator > 0.0)) {
                    phi[i] = 0.0;
                } else {
                    const double root = std::sqrt(denominator * denominator + 4.0 * a2 * numerator);
                    phi[i] = denominator >= 0.0 ? 2.0 * numerator / (denominator + root)
                                                : (root - denominator) / (2.0 * a2);
                }
                continue;
            }
        }
        phi[i] = numerator / denominator;
    }
}

} // namespace

void point_implicit_update(const std::vector<double>& nodal_mass,
                           const std::vector<double>& elcoe2,
                           const std::vector<double>& rhs,
                           const std::vector<double>& production,
                           const std::vector<double>& destruction,
                           bool enforce_positivity,
                           std::vector<double>& phi) {
    update<false>(nodal_mass, elcoe2, rhs, production, destruction, nullptr, enforce_positivity, phi);
}

void point_implicit_update(const std::vector<double>& nodal_mass,
                           const std::vector<double>& elcoe2,
                           const std::vector<double>& rhs,
                           const std::vector<double>& production,
                           const std::vector<double>& destruction,
                           const std::vector<double>& quadratic,
                           bool enforce_positivity,
                           std::vector<double>& phi) {
    update<true>(nodal_mass, elcoe2, rhs, production, destruction, &quadratic, enforce_positivity, phi);
}

} // namespace hydrama
