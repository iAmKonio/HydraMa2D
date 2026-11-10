#include "geometry/FieldOps.hpp"

#include <algorithm>

namespace hydrama {

std::vector<double> lumped_nodal_mass(const Mesh& mesh) {
    std::vector<double> mass(mesh.npoin, 0.0);
    for (int ie = 0; ie < mesh.nelem; ++ie) {
        const double third = mesh.detJ[ie] / 6.0; // area / 3
        for (int k = 0; k < Mesh::nep; ++k) mass[mesh.intma[ie][k]] += third;
    }
    return mass;
}

void compute_scalar_gradients(const Mesh& mesh, const std::vector<double>& phi,
                              std::vector<std::array<double, 2>>& grad) {
    grad.resize(mesh.nelem);
    for (int ie = 0; ie < mesh.nelem; ++ie) {
        const auto& n = mesh.intma[ie];
        const double* dNdx = mesh.dNkdx[ie].dNdx;
        const double* dNdy = mesh.dNkdx[ie].dNdy;
        const double p0 = phi[n[0]], p1 = phi[n[1]], p2 = phi[n[2]];
        grad[ie] = {dNdx[0] * p0 + dNdx[1] * p1 + dNdx[2] * p2,
                    dNdy[0] * p0 + dNdy[1] * p1 + dNdy[2] * p2};
    }
}

void compute_velocity_gradients(const Mesh& mesh, const std::vector<double>& u,
                                const std::vector<double>& v, std::vector<VelocityGradient>& grad) {
    grad.resize(mesh.nelem);
    for (int ie = 0; ie < mesh.nelem; ++ie) {
        const auto& n = mesh.intma[ie];
        const double* dNdx = mesh.dNkdx[ie].dNdx;
        const double* dNdy = mesh.dNkdx[ie].dNdy;
        const double u0 = u[n[0]], u1 = u[n[1]], u2 = u[n[2]];
        const double v0 = v[n[0]], v1 = v[n[1]], v2 = v[n[2]];
        VelocityGradient& g = grad[ie];
        g.dudx = dNdx[0] * u0 + dNdx[1] * u1 + dNdx[2] * u2;
        g.dudy = dNdy[0] * u0 + dNdy[1] * u1 + dNdy[2] * u2;
        g.dvdx = dNdx[0] * v0 + dNdx[1] * v1 + dNdx[2] * v2;
        g.dvdy = dNdy[0] * v0 + dNdy[1] * v1 + dNdy[2] * v2;
    }
}

void project_to_nodes(const Mesh& mesh, const std::vector<double>& nodal_mass,
                      const std::vector<double>& elem_values, std::vector<double>& nodal_values) {
    nodal_values.assign(mesh.npoin, 0.0);
    for (int ie = 0; ie < mesh.nelem; ++ie) {
        const double w = elem_values[ie] * mesh.detJ[ie] / 6.0;
        for (int k = 0; k < Mesh::nep; ++k) nodal_values[mesh.intma[ie][k]] += w;
    }
    for (int ip = 0; ip < mesh.npoin; ++ip) nodal_values[ip] /= nodal_mass[ip];
}

void element_mean(const Mesh& mesh, const std::vector<double>& nodal_values,
                  std::vector<double>& elem_values) {
    elem_values.resize(mesh.nelem);
    for (int ie = 0; ie < mesh.nelem; ++ie) {
        const auto& n = mesh.intma[ie];
        elem_values[ie] = (nodal_values[n[0]] + nodal_values[n[1]] + nodal_values[n[2]]) / 3.0;
    }
}

// ---------------------------------------------------------------------------
// Recovered nodal gradients
// ---------------------------------------------------------------------------

std::string_view to_string(GradientRecovery method) noexcept {
    return method == GradientRecovery::PolynomialPreserving ? "ppr" : "lumped-projection";
}

std::optional<GradientRecovery> parse_gradient_recovery(std::string_view name) {
    if (name == "ppr" || name == "polynomial-preserving") return GradientRecovery::PolynomialPreserving;
    if (name == "lumped-projection" || name == "lumped") return GradientRecovery::LumpedProjection;
    return std::nullopt;
}

namespace {

constexpr int kQuadTerms = 6; // 1, x, y, x^2, xy, y^2

// Least-squares quadratic fit centred at (xi, yi) through the nodal values of
// `patch`; on success fills the derivative weights (d/dx, d/dy at the centre)
// per patch node and returns true. Coordinates are scaled by the patch radius;
// the 6x6 normal equations are solved by Gaussian elimination with partial
// pivoting and rejected if a pivot is tiny (fewer than 6 points, points on a
// conic, ...).
bool ppr_weights(const Mesh& mesh, double xi, double yi, const std::vector<int>& patch,
                 std::vector<double>& wx, std::vector<double>& wy) {
    const int n = static_cast<int>(patch.size());
    if (n < kQuadTerms) return false;
    double scale = 0.0;
    for (int j : patch) scale = std::max({scale, std::abs(mesh.x[j] - xi), std::abs(mesh.y[j] - yi)});
    if (!(scale > 0.0)) return false;

    auto basis = [&](int j, double q[kQuadTerms]) {
        const double dx = (mesh.x[j] - xi) / scale, dy = (mesh.y[j] - yi) / scale;
        q[0] = 1.0;
        q[1] = dx;
        q[2] = dy;
        q[3] = dx * dx;
        q[4] = dx * dy;
        q[5] = dy * dy;
    };

    // Augmented system [M | e1 e2]: columns 1 and 2 of M^-1 (= rows, M symmetric).
    double a[kQuadTerms][kQuadTerms + 2] = {};
    for (int j : patch) {
        double q[kQuadTerms];
        basis(j, q);
        for (int r = 0; r < kQuadTerms; ++r)
            for (int c = 0; c < kQuadTerms; ++c) a[r][c] += q[r] * q[c];
    }
    a[1][kQuadTerms] = 1.0;
    a[2][kQuadTerms + 1] = 1.0;

    double max_diag = 0.0;
    for (int r = 0; r < kQuadTerms; ++r) max_diag = std::max(max_diag, a[r][r]);
    for (int col = 0; col < kQuadTerms; ++col) {
        int piv = col;
        for (int r = col + 1; r < kQuadTerms; ++r)
            if (std::abs(a[r][col]) > std::abs(a[piv][col])) piv = r;
        if (!(std::abs(a[piv][col]) > 1.0e-10 * max_diag)) return false;
        if (piv != col)
            for (int c = 0; c < kQuadTerms + 2; ++c) std::swap(a[piv][c], a[col][c]);
        for (int r = 0; r < kQuadTerms; ++r) {
            if (r == col) continue;
            const double f = a[r][col] / a[col][col];
            if (f == 0.0) continue;
            for (int c = col; c < kQuadTerms + 2; ++c) a[r][c] -= f * a[col][c];
        }
    }
    double zx[kQuadTerms], zy[kQuadTerms];
    for (int r = 0; r < kQuadTerms; ++r) {
        zx[r] = a[r][kQuadTerms] / a[r][r];
        zy[r] = a[r][kQuadTerms + 1] / a[r][r];
    }

    wx.resize(n);
    wy.resize(n);
    for (int k = 0; k < n; ++k) {
        double q[kQuadTerms];
        basis(patch[k], q);
        double sx = 0.0, sy = 0.0;
        for (int r = 0; r < kQuadTerms; ++r) {
            sx += zx[r] * q[r];
            sy += zy[r] * q[r];
        }
        wx[k] = sx / scale;
        wy[k] = sy / scale;
    }
    return true;
}

} // namespace

NodalGradient::NodalGradient(const Mesh& mesh, GradientRecovery method) : method_(method) {
    const int npoin = mesh.npoin;

    // Node -> element adjacency (CSR).
    std::vector<int> adj_offset(npoin + 1, 0);
    for (const auto& e : mesh.intma)
        for (int ip : e) ++adj_offset[ip + 1];
    for (int ip = 0; ip < npoin; ++ip) adj_offset[ip + 1] += adj_offset[ip];
    std::vector<int> adj(adj_offset[npoin]);
    {
        std::vector<int> fill(adj_offset.begin(), adj_offset.end() - 1);
        for (int ie = 0; ie < mesh.nelem; ++ie)
            for (int ip : mesh.intma[ie]) adj[fill[ip]++] = ie;
    }

    std::vector<char> on_boundary(npoin, 0);
    for (const auto& s : mesh.iside) {
        on_boundary[s.node1] = 1;
        on_boundary[s.node2] = 1;
    }
    const std::vector<double> mass = lumped_nodal_mass(mesh);

    offset_.assign(1, 0);
    node_.clear();
    gx_.clear();
    gy_.clear();
    fallback_ = 0;

    std::vector<char> mark(npoin, 0);
    std::vector<int> ring1, ring2, list;
    std::vector<double> acc_x(npoin, 0.0), acc_y(npoin, 0.0), wx, wy;

    // Nodes of the elements around every node in `seeds` (marked, no duplicates).
    auto collect = [&](const std::vector<int>& seeds, std::vector<int>& out) {
        out.clear();
        for (int s : seeds) {
            for (int k = adj_offset[s]; k < adj_offset[s + 1]; ++k) {
                for (int jp : mesh.intma[adj[k]]) {
                    if (!mark[jp]) {
                        mark[jp] = 1;
                        out.push_back(jp);
                    }
                }
            }
        }
        for (int jp : out) mark[jp] = 0;
    };

    for (int ip = 0; ip < npoin; ++ip) {
        bool done = false;
        if (method == GradientRecovery::PolynomialPreserving) {
            collect({ip}, ring1); // the node and its neighbours
            const std::vector<int>* patch = &ring1;
            done = !on_boundary[ip] && ppr_weights(mesh, mesh.x[ip], mesh.y[ip], ring1, wx, wy);
            if (!done) {
                collect(ring1, ring2); // two rings
                patch = &ring2;
                done = ppr_weights(mesh, mesh.x[ip], mesh.y[ip], ring2, wx, wy);
            }
            if (done) {
                node_.insert(node_.end(), patch->begin(), patch->end());
                gx_.insert(gx_.end(), wx.begin(), wx.end());
                gy_.insert(gy_.end(), wy.begin(), wy.end());
            } else {
                ++fallback_;
            }
        }
        if (!done) {
            // Lumped projection of the element gradients as a nodal stencil:
            // G_ij = sum_{e contains i, j} (A_e/3) / M_i grad N_j^e.
            list.clear();
            for (int k = adj_offset[ip]; k < adj_offset[ip + 1]; ++k) {
                const int ie = adj[k];
                const double w = mesh.detJ[ie] / 6.0 / mass[ip];
                for (int c = 0; c < Mesh::nep; ++c) {
                    const int jp = mesh.intma[ie][c];
                    if (!mark[jp]) {
                        mark[jp] = 1;
                        list.push_back(jp);
                    }
                    acc_x[jp] += w * mesh.dNkdx[ie].dNdx[c];
                    acc_y[jp] += w * mesh.dNkdx[ie].dNdy[c];
                }
            }
            for (int jp : list) {
                node_.push_back(jp);
                gx_.push_back(acc_x[jp]);
                gy_.push_back(acc_y[jp]);
                acc_x[jp] = acc_y[jp] = 0.0;
                mark[jp] = 0;
            }
        }
        offset_.push_back(static_cast<int>(node_.size()));
    }
}

void NodalGradient::gradient(const std::vector<double>& phi, std::vector<std::array<double, 2>>& grad) const {
    const int n = npoin();
    grad.resize(n);
    for (int ip = 0; ip < n; ++ip) {
        double sx = 0.0, sy = 0.0;
        for (int k = offset_[ip]; k < offset_[ip + 1]; ++k) {
            const double p = phi[node_[k]];
            sx += gx_[k] * p;
            sy += gy_[k] * p;
        }
        grad[ip] = {sx, sy};
    }
}

void NodalGradient::velocity_gradient(const std::vector<double>& u, const std::vector<double>& v,
                                      std::vector<VelocityGradient>& grad) const {
    const int n = npoin();
    grad.resize(n);
    for (int ip = 0; ip < n; ++ip) {
        VelocityGradient g;
        for (int k = offset_[ip]; k < offset_[ip + 1]; ++k) {
            const int jp = node_[k];
            g.dudx += gx_[k] * u[jp];
            g.dudy += gy_[k] * u[jp];
            g.dvdx += gx_[k] * v[jp];
            g.dvdy += gy_[k] * v[jp];
        }
        grad[ip] = g;
    }
}

// ---------------------------------------------------------------------------
// Kinematics
// ---------------------------------------------------------------------------

void FlowKinematics::compute(const Mesh& mesh, const NodalGradient& recovery,
                             const std::vector<double>& u, const std::vector<double>& v) {
    compute_velocity_gradients(mesh, u, v, grad_elem);

    strain_elem.resize(mesh.nelem);
    vorticity_elem.resize(mesh.nelem);
    for (int ie = 0; ie < mesh.nelem; ++ie) {
        strain_elem[ie] = strain_rate_magnitude(grad_elem[ie]);
        vorticity_elem[ie] = vorticity_magnitude(grad_elem[ie]);
    }

    recovery.velocity_gradient(u, v, grad_node);
    strain_node.resize(mesh.npoin);
    vorticity_node.resize(mesh.npoin);
    for (int ip = 0; ip < mesh.npoin; ++ip) {
        strain_node[ip] = strain_rate_magnitude(grad_node[ip]);
        vorticity_node[ip] = vorticity_magnitude(grad_node[ip]);
    }
}

} // namespace hydrama
