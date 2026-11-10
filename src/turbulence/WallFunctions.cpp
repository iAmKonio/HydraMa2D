#include "turbulence/WallFunctions.hpp"

#include <algorithm>
#include <cmath>

namespace hydrama::wallfn {

double y_plus_laminar(const LogLaw& law) {
    // y = ln(E y) / kappa; the map is a contraction near the root (derivative 1/(kappa y) ~ 0.2).
    double y = 11.0;
    for (int it = 0; it < 200; ++it) {
        const double next = std::log(std::max(law.E * y, 1.0 + 1e-12)) / law.kappa;
        if (std::abs(next - y) <= 1e-15 * y) return next;
        y = next;
    }
    return y;
}

double u_plus(double y_plus, const LogLaw& law) {
    const double lam = y_plus_laminar(law);
    return y_plus <= lam ? y_plus : std::log(law.E * y_plus) / law.kappa;
}

double nut_wall(double y_plus, double nu, double y_plus_lam, const LogLaw& law) {
    if (!(y_plus > y_plus_lam)) return 0.0;
    return nu * std::max(y_plus * law.kappa / std::log(law.E * y_plus) - 1.0, 0.0);
}

double friction_velocity(double k, double c_mu) { return std::pow(c_mu, 0.25) * std::sqrt(std::max(k, 0.0)); }

double epsilon_wall(double k, double y_p, double c_mu, const LogLaw& law) {
    const double kk = std::max(k, 0.0);
    return std::pow(c_mu, 0.75) * kk * std::sqrt(kk) / (law.kappa * y_p);
}

} // namespace hydrama::wallfn
