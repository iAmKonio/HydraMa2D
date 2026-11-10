#pragma once

// Element and nodal field operations on linear (P1) triangles.
//
//  * Gradients of P1 fields are constant per element:
//      grad(phi)_e = sum_k grad(N_k) phi_k.
//  * Velocity gradient tensor per element (du/dx, du/dy, dv/dx, dv/dy), the
//    strain-rate magnitude  S = sqrt(2 S_ij S_ij)
//                             = sqrt(2 (du/dx)^2 + 2 (dv/dy)^2 + (du/dy + dv/dx)^2)
//    and the vorticity magnitude  Omega = sqrt(2 W_ij W_ij) = |dv/dx - du/dy|,
//    with S_ij = (du_i/dx_j + du_j/dx_i)/2, W_ij = (du_i/dx_j - du_j/dx_i)/2.
//  * Lumped L2 projection of element values to the nodes (area-weighted
//    average): f_i = sum_e (A_e/3) f_e / M_i,  M_i = sum_e A_e/3.
//  * Recovered nodal gradients of nodal fields (NodalGradient), stored as
//    precomputed sparse nodal stencils  grad(phi)_i = sum_j (Gx_ij, Gy_ij) phi_j,
//    built once for the static mesh. Two methods:
//    - polynomial preserving recovery (PPR, Zhang & Naga 2005, default): a
//      quadratic polynomial is least-squares fitted to the nodal values of
//      the node patch (one ring of neighbours for interior nodes, two rings
//      for boundary nodes or ill-conditioned patches) and differentiated at
//      the node. Exact for quadratic fields on arbitrary meshes, i.e.
//      second-order nodal gradients everywhere, including boundaries.
//      Nodes where no fit is possible (tiny meshes) fall back to:
//    - lumped L2 projection of the element gradients. Second order only on
//      symmetric patches of uniform meshes; first order at boundary nodes
//      and on irregular meshes.
//  * Nodal S and Omega are computed from the recovered nodal velocity
//    gradient, not by recovering the magnitudes. This keeps Omega = 0 where
//    the vorticity changes sign (e.g. a channel centre line).
//
// Reference: Z. Zhang, A. Naga, "A new finite element gradient recovery
// method: superconvergence property", SIAM J. Sci. Comput. 26(4) (2005)
// 1192-1213.

#include "Mesh.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <string_view>
#include <vector>

namespace hydrama {

struct VelocityGradient {
    double dudx = 0.0, dudy = 0.0;
    double dvdx = 0.0, dvdy = 0.0;
};

// sqrt(2 S_ij S_ij)
inline double strain_rate_magnitude(const VelocityGradient& g) noexcept {
    const double shear = g.dudy + g.dvdx;
    return std::sqrt(2.0 * (g.dudx * g.dudx + g.dvdy * g.dvdy) + shear * shear);
}

// sqrt(2 W_ij W_ij) = |omega_z|
inline double vorticity_magnitude(const VelocityGradient& g) noexcept {
    return std::abs(g.dvdx - g.dudy);
}

// Lumped nodal mass M_i = sum_e A_e / 3 (requires detJ).
std::vector<double> lumped_nodal_mass(const Mesh& mesh);

// Element gradients of a nodal scalar field: grad[e] = {dphi/dx, dphi/dy}.
void compute_scalar_gradients(const Mesh& mesh, const std::vector<double>& phi,
                              std::vector<std::array<double, 2>>& grad);

// Element velocity gradient tensors.
void compute_velocity_gradients(const Mesh& mesh, const std::vector<double>& u,
                                const std::vector<double>& v, std::vector<VelocityGradient>& grad);

// Lumped L2 projection of element-constant values to the nodes.
void project_to_nodes(const Mesh& mesh, const std::vector<double>& nodal_mass,
                      const std::vector<double>& elem_values, std::vector<double>& nodal_values);

// Element mean of a nodal field (exact element integral / area for P1).
void element_mean(const Mesh& mesh, const std::vector<double>& nodal_values,
                  std::vector<double>& elem_values);

enum class GradientRecovery {
    PolynomialPreserving, // PPR (default)
    LumpedProjection,     // area-weighted average of element gradients
};

std::string_view to_string(GradientRecovery method) noexcept;
// "ppr" | "polynomial-preserving" | "lumped-projection" | "lumped"
std::optional<GradientRecovery> parse_gradient_recovery(std::string_view name);

// Recovered nodal gradient operator (see above).
class NodalGradient {
public:
    NodalGradient() = default;
    NodalGradient(const Mesh& mesh, GradientRecovery method); // requires detJ and dNkdx

    GradientRecovery method() const noexcept { return method_; }
    int npoin() const noexcept { return static_cast<int>(offset_.empty() ? 0 : offset_.size() - 1); }
    // Nodes where PPR was impossible and the lumped projection is used.
    int fallback_nodes() const noexcept { return fallback_; }

    // Stencil of node i: nodes()[k], coefficients gx()[k], gy()[k] for
    // k in [offsets()[i], offsets()[i+1]).
    const std::vector<int>& offsets() const noexcept { return offset_; }
    const std::vector<int>& nodes() const noexcept { return node_; }
    const std::vector<double>& gx() const noexcept { return gx_; }
    const std::vector<double>& gy() const noexcept { return gy_; }

    void gradient(const std::vector<double>& phi, std::vector<std::array<double, 2>>& grad) const;
    void velocity_gradient(const std::vector<double>& u, const std::vector<double>& v,
                           std::vector<VelocityGradient>& grad) const;

private:
    GradientRecovery method_ = GradientRecovery::PolynomialPreserving;
    std::vector<int> offset_, node_;
    std::vector<double> gx_, gy_;
    int fallback_ = 0;
};

// Velocity-derived quantities shared by all turbulence models: one element
// pass plus one sparse stencil pass. Buffers are reused between calls.
struct FlowKinematics {
    std::vector<VelocityGradient> grad_elem; // per element
    std::vector<double> strain_elem;         // S per element
    std::vector<double> vorticity_elem;      // Omega per element
    std::vector<VelocityGradient> grad_node; // recovered nodal gradient
    std::vector<double> strain_node;         // S from grad_node
    std::vector<double> vorticity_node;      // Omega from grad_node

    void compute(const Mesh& mesh, const NodalGradient& recovery,
                 const std::vector<double>& u, const std::vector<double>& v);
};

} // namespace hydrama
