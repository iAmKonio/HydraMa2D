#include "turbulence/TurbulenceBC.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace hydrama {

namespace {

DirichletBuilder inlet_builder(const Mesh& mesh, const TurbulenceConfig& config,
                               const std::function<double(int flag)>& inlet_value) {
    DirichletBuilder builder(mesh.npoin);
    std::map<int, double> inlet_cache; // one evaluation per flag
    for (const BoundarySide& side : mesh.iside) {
        if (config.boundary_kind(side.flag) != BoundaryKind::Inlet) continue;
        auto it = inlet_cache.find(side.flag);
        if (it == inlet_cache.end()) it = inlet_cache.emplace(side.flag, inlet_value(side.flag)).first;
        builder.set(side.node1, it->second);
        builder.set(side.node2, it->second);
    }
    return builder;
}

} // namespace

DirichletSet build_turbulence_dirichlet(const Mesh& mesh,
                                        const TurbulenceConfig& config,
                                        const std::function<double(int flag)>& inlet_value,
                                        const std::function<double(int node)>& wall_value) {
    DirichletBuilder builder = inlet_builder(mesh, config, inlet_value);
    for (const BoundarySide& side : mesh.iside) {
        if (config.boundary_kind(side.flag) != BoundaryKind::Wall) continue;
        builder.set(side.node1, wall_value(side.node1));
        builder.set(side.node2, wall_value(side.node2));
    }
    return builder.build();
}

TwoEquationInlet two_equation_inlet(const InletTurbulence& in, double nu, double reference_velocity) {
    constexpr double c_mu = 0.09;
    constexpr double tiny = 1.0e-30;
    TwoEquationInlet r;
    if (in.k) {
        r.k = *in.k;
    } else {
        const double u = reference_velocity * in.intensity;
        r.k = 1.5 * u * u;
    }
    if (in.epsilon) {
        r.epsilon = *in.epsilon;
    } else if (in.omega) {
        r.epsilon = c_mu * r.k * *in.omega;
    } else if (in.viscosity_ratio && *in.viscosity_ratio > 0.0) {
        r.epsilon = c_mu * r.k * r.k / (*in.viscosity_ratio * nu);
    } else {
        r.epsilon = std::pow(c_mu, 0.75) * std::pow(r.k, 1.5) / in.length_scale;
    }
    r.epsilon = std::max(r.epsilon, tiny);
    r.omega = in.omega ? *in.omega : r.epsilon / (c_mu * std::max(r.k, tiny));
    r.omega = std::max(r.omega, tiny);
    return r;
}

DirichletSet build_turbulence_dirichlet(const Mesh& mesh,
                                        const TurbulenceConfig& config,
                                        const std::function<double(int flag)>& inlet_value,
                                        std::optional<double> wall_value) {
    DirichletBuilder builder(mesh.npoin);
    std::map<int, double> inlet_cache; // one evaluation per flag

    for (const BoundarySide& side : mesh.iside) {
        if (config.boundary_kind(side.flag) != BoundaryKind::Inlet) continue;
        auto it = inlet_cache.find(side.flag);
        if (it == inlet_cache.end()) it = inlet_cache.emplace(side.flag, inlet_value(side.flag)).first;
        builder.set(side.node1, it->second);
        builder.set(side.node2, it->second);
    }

    if (wall_value) {
        for (const BoundarySide& side : mesh.iside) {
            if (config.boundary_kind(side.flag) != BoundaryKind::Wall) continue;
            builder.set(side.node1, *wall_value);
            builder.set(side.node2, *wall_value);
        }
    }
    return builder.build();
}

} // namespace hydrama
