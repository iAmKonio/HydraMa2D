#include "turbulence/KEpsilon.hpp"

#include "turbulence/TurbulenceBC.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

namespace hydrama {

namespace ke {

double turbulence_reynolds(double k, double e, double nu) noexcept {
    if (!(k > 0.0)) return 0.0;
    return k * k / (nu * std::max(e, kTiny));
}

} // namespace ke

namespace {

// |t . (G n)|: tangential component of the normal derivative of the velocity.
double wall_shear_rate(const VelocityGradient& g, double nx, double ny) {
    const double gnx = g.dudx * nx + g.dudy * ny;
    const double gny = g.dvdx * nx + g.dvdy * ny;
    return std::abs(ny * gnx - nx * gny);
}

} // namespace

template <class Variant, class Damping>
KEpsilon<Variant, Damping>::KEpsilon(TurbulenceConfig config) : config_(std::move(config)) {}

template <class Variant, class Damping>
TwoEquationInlet KEpsilon<Variant, Damping>::inlet_values(int flag, double nu) const {
    return two_equation_inlet(config_.inlet_for(flag), nu, config_.reference_velocity);
}

template <class Variant, class Damping>
TwoEquationInlet KEpsilon<Variant, Damping>::freestream_values(double nu) const {
    return two_equation_inlet(config_.inlet, nu, config_.reference_velocity);
}

template <class Variant, class Damping>
void KEpsilon<Variant, Damping>::initialize_fields(const TurbulenceContext& ctx) {
    const int npoin = ctx.mesh.npoin;
    const TwoEquationInlet fs = freestream_values(ctx.nu);
    this->phi_[0].assign(npoin, fs.k);
    this->phi_[1].assign(npoin, fs.epsilon);

    auto k_in = [&](int flag) { return inlet_values(flag, ctx.nu).k; };
    auto e_in = [&](int flag) { return inlet_values(flag, ctx.nu).epsilon; };
    if constexpr (Damping::kLowRe) {
        // Resolved wall: k = 0, eps~ = 0.
        this->bcs_[0] = build_turbulence_dirichlet(ctx.mesh, config_, k_in, std::optional<double>(0.0));
        this->bcs_[1] = build_turbulence_dirichlet(ctx.mesh, config_, e_in, std::optional<double>(0.0));
    } else {
        // Wall functions: k natural at walls, eps set by apply_dynamic_bcs().
        this->bcs_[0] = build_turbulence_dirichlet(ctx.mesh, config_, k_in, std::nullopt);
        this->bcs_[1] = build_turbulence_dirichlet(ctx.mesh, config_, e_in, std::nullopt);
        y_plus_lam_ = wallfn::y_plus_laminar(law_);

        // Distinct wall-adjacent elements and the number of wall faces of each.
        const WallGeometry& w = ctx.walls;
        std::vector<int> index(ctx.mesh.nelem, -1);
        wall_elems_.clear();
        wall_elem_faces_.clear();
        face_wall_elem_.assign(w.num_faces(), -1);
        for (int f = 0; f < w.num_faces(); ++f) {
            const int e = w.face(f).elem;
            if (index[e] < 0) {
                index[e] = static_cast<int>(wall_elems_.size());
                wall_elems_.push_back(e);
                wall_elem_faces_.push_back(0);
            }
            face_wall_elem_[f] = index[e];
            ++wall_elem_faces_[index[e]];
        }
        wall_elem_sum_.assign(wall_elems_.size(), 0.0);

        // Wall-function layer: every node of an element that touches the wall
        // (contains a wall node; its velocity gradient sees the no-slip value),
        // associated with the wall faces of those wall nodes.
        std::vector<std::vector<int>> faces_of(ctx.mesh.npoin);
        for (int ie = 0; ie < ctx.mesh.nelem; ++ie) {
            const ElementNodes e = ctx.mesh.element_nodes(ie);
            for (int wn : e) {
                const int iw = w.wall_index(wn);
                if (iw < 0) continue;
                for (int ip : e)
                    for (int f : w.wall_node_faces(iw)) faces_of[ip].push_back(f);
            }
        }
        for (int ip = 0; ip < ctx.mesh.npoin; ++ip) {
            std::vector<int>& list = faces_of[ip];
            const int iw = w.wall_index(ip);
            if (iw >= 0) { // wall nodes: their own faces
                const auto own = w.wall_node_faces(iw);
                list.assign(own.begin(), own.end());
            }
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
        }
        layer_nodes_.clear();
        layer_faces_.clear();
        layer_height_.clear();
        layer_offsets_.assign(1, 0);
        for (int ip = 0; ip < ctx.mesh.npoin; ++ip) {
            if (faces_of[ip].empty()) continue;
            layer_nodes_.push_back(ip);
            layer_faces_.insert(layer_faces_.end(), faces_of[ip].begin(), faces_of[ip].end());
            layer_offsets_.push_back(static_cast<int>(layer_faces_.size()));
            const int iw = w.wall_index(ip);
            layer_height_.push_back(iw >= 0 ? w.wall_node_yp(iw) : w.wall_distance()[ip]);
        }
    }
    D_.assign(npoin, 0.0);
    E_.assign(npoin, 0.0);
}

template <class Variant, class Damping>
void KEpsilon<Variant, Damping>::apply_dynamic_bcs(const TurbulenceContext& ctx) {
    if constexpr (!Damping::kLowRe) {
        const WallGeometry& w = ctx.walls;
        std::vector<double>& k = this->phi_[0];
        std::vector<double>& e = this->phi_[1];
        for (int iw = 0; iw < w.num_wall_nodes(); ++iw) {
            const int ip = w.wall_nodes()[iw];
            e[ip] = std::max(wallfn::epsilon_wall(k[ip], w.wall_node_yp(iw), Variant::c_mu, law_), ke::kTiny);
        }
    } else {
        (void)ctx;
    }
}

template <class Variant, class Damping>
void KEpsilon<Variant, Damping>::compute_eddy_viscosity(const TurbulenceContext& ctx) {
    const double cap = config_.nut_ratio_max * ctx.nu;
    const std::vector<double>& k = this->phi_[0];
    const std::vector<double>& e = this->phi_[1];
    for (int ip = 0; ip < ctx.mesh.npoin; ++ip) {
        this->nut_[ip] = std::min(ke::eddy_viscosity<Variant, Damping>(k[ip], e[ip], ctx.nu), cap);
    }
}

template <class Variant, class Damping>
void KEpsilon<Variant, Damping>::adjust_element_viscosity(const TurbulenceContext& ctx) {
    if constexpr (!Damping::kLowRe) {
        // Wall-adjacent elements carry the log-law eddy viscosity (mean over
        // their wall faces), so that Step 1 produces the log-law wall shear.
        const WallGeometry& w = ctx.walls;
        const std::vector<double>& k = this->phi_[0];
        const double cap = config_.nut_ratio_max * ctx.nu;
        std::fill(wall_elem_sum_.begin(), wall_elem_sum_.end(), 0.0);
        for (int f = 0; f < w.num_faces(); ++f) {
            const WallFace& face = w.face(f);
            const double kf = 0.5 * (k[face.node1] + k[face.node2]);
            const double y_plus = wallfn::friction_velocity(kf, Variant::c_mu) * face.yp / ctx.nu;
            wall_elem_sum_[face_wall_elem_[f]] += std::min(wallfn::nut_wall(y_plus, ctx.nu, y_plus_lam_, law_), cap);
        }
        for (std::size_t j = 0; j < wall_elems_.size(); ++j) {
            this->nut_elem_[wall_elems_[j]] = wall_elem_sum_[j] / wall_elem_faces_[j];
        }
    } else {
        (void)ctx;
    }
}

template <class Variant, class Damping>
void KEpsilon<Variant, Damping>::compute_sources(const TurbulenceContext& ctx) {
    const Mesh& mesh = ctx.mesh;
    const int npoin = mesh.npoin;
    const std::vector<double>& k = this->phi_[0];
    const std::vector<double>& e = this->phi_[1];
    const std::vector<double>& nut = this->nut_;
    const std::vector<double>& strain = this->kinematics_.strain_node;

    if constexpr (Damping::kLowRe) {
        // D = 2 nu |grad sqrt(k)|^2 from the recovered gradient of sqrt(k).
        sqrt_k_.resize(npoin);
        for (int ip = 0; ip < npoin; ++ip) sqrt_k_[ip] = std::sqrt(std::max(k[ip], 0.0));
        this->recovery_.gradient(sqrt_k_, grad_);
        for (int ip = 0; ip < npoin; ++ip) D_[ip] = 2.0 * ctx.nu * (grad_[ip][0] * grad_[ip][0] + grad_[ip][1] * grad_[ip][1]);
        // E = 2 nu nu_t sum (d2 u_i / dx_j dx_k)^2: recovery applied to the
        // recovered nodal velocity gradient (second derivatives vanish per P1 element).
        const auto& g = this->kinematics_.grad_node;
        std::vector<double> comp(npoin);
        for (int c = 0; c < 4; ++c) {
            for (int ip = 0; ip < npoin; ++ip) {
                comp[ip] = c == 0 ? g[ip].dudx : c == 1 ? g[ip].dudy : c == 2 ? g[ip].dvdx : g[ip].dvdy;
            }
            this->recovery_.gradient(comp, second_[c]);
        }
        for (int ip = 0; ip < npoin; ++ip) {
            double sum = 0.0;
            for (int c = 0; c < 4; ++c) sum += second_[c][ip][0] * second_[c][ip][0] + second_[c][ip][1] * second_[c][ip][1];
            E_[ip] = 2.0 * ctx.nu * nut[ip] * sum;
        }
    }

    // Wall-function production in the wall-function layer:
    // P = tau_w u_tau / (kappa y), tau_w = (nu + nu_t,e) |t . (grad u) n| of the
    // wall elements (length-weighted over the faces whose element contains the node).
    std::vector<double> wall_production;
    if constexpr (!Damping::kLowRe) {
        const WallGeometry& w = ctx.walls;
        std::vector<double> tau_face(w.num_faces());
        for (int f = 0; f < w.num_faces(); ++f) {
            const WallFace& face = w.face(f);
            tau_face[f] = (ctx.nu + this->nut_elem_[face.elem]) *
                          wall_shear_rate(this->kinematics_.grad_elem[face.elem], face.nx, face.ny);
        }
        wall_production.assign(npoin, -1.0);
        for (std::size_t j = 0; j < layer_nodes_.size(); ++j) {
            double tau = 0.0, len = 0.0;
            for (int q = layer_offsets_[j]; q < layer_offsets_[j + 1]; ++q) {
                const WallFace& face = w.face(layer_faces_[q]);
                tau += tau_face[layer_faces_[q]] * face.length;
                len += face.length;
            }
            const int ip = layer_nodes_[j];
            const double u_tau = wallfn::friction_velocity(k[ip], Variant::c_mu);
            wall_production[ip] = (tau / len) * u_tau / (law_.kappa * std::max(layer_height_[j], ke::kTiny));
        }
    }

    for (int ip = 0; ip < npoin; ++ip) {
        const ke::PointState s{k[ip], e[ip], ctx.nu, nut[ip], strain[ip], D_[ip], E_[ip]};
        const double override_p = wall_production.empty() ? -1.0 : wall_production[ip];
        const ke::SourceTerms t = ke::source<Variant, Damping>(s, config_.production_limiter, override_p);
        this->production_[0][ip] = t.k.P;
        this->destruction_[0][ip] = t.k.D;
        this->production_[1][ip] = t.e.P;
        this->destruction_[1][ip] = t.e.D;
        this->quadratic_[1][ip] = t.e.B;
    }
}

template <class Variant, class Damping>
void KEpsilon<Variant, Damping>::compute_diffusivity(const TurbulenceContext& ctx) {
    const int npoin = ctx.mesh.npoin;
    diffusivity_node_k_.resize(npoin);
    diffusivity_node_e_.resize(npoin);
    for (int ip = 0; ip < npoin; ++ip) {
        diffusivity_node_k_[ip] = ctx.nu + this->nut_[ip] / Variant::sigma_k;
        diffusivity_node_e_[ip] = ctx.nu + this->nut_[ip] / Variant::sigma_eps;
    }
    element_mean(ctx.mesh, diffusivity_node_k_, this->diffusivity_elem_[0]);
    element_mean(ctx.mesh, diffusivity_node_e_, this->diffusivity_elem_[1]);
}

template class KEpsilon<ke::Standard, ke::NoDamping>;
template class KEpsilon<ke::Standard, ke::LaunderSharma>;

} // namespace hydrama
