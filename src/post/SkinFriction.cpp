#include "post/SkinFriction.hpp"
#include "BoundaryKind.hpp"
#include "mesh/BoundaryNaming.hpp"
#include "numerics/QuadElement.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace hydrama::post {

namespace {

// Velocity gradient at the midpoint of side (n1, n2) of element e
struct SideGradient {
    double ux = 0.0, uy = 0.0, vx = 0.0, vy = 0.0;
};

SideGradient compute_side_gradient(const Mesh& mesh, int e, int n1, int n2,
                                   const std::vector<double>& u,
                                   const std::vector<double>& v) {
    SideGradient g;
    const ElementNodes nodes = mesh.element_nodes(e);
    if (!mesh.is_quad(e)) {
        const ElementDeriv& d = mesh.dNkdx[static_cast<std::size_t>(e)];
        for (int a = 0; a < 3; ++a) {
            const auto ip = static_cast<std::size_t>(nodes[a]);
            g.ux += u[ip] * d.dNdx[a];
            g.uy += u[ip] * d.dNdy[a];
            g.vx += v[ip] * d.dNdx[a];
            g.vy += v[ip] * d.dNdy[a];
        }
        return g;
    }

    double x[4], y[4];
    double xi = 0.0, eta = 0.0;
    int on_side = 0;
    for (int a = 0; a < 4; ++a) {
        x[a] = mesh.x[static_cast<std::size_t>(nodes[a])];
        y[a] = mesh.y[static_cast<std::size_t>(nodes[a])];
        if (nodes[a] == n1 || nodes[a] == n2) {
            xi += 0.5 * q1::kNodeXi[a];
            eta += 0.5 * q1::kNodeEta[a];
            ++on_side;
        }
    }
    if (on_side != 2) {
        throw std::invalid_argument("compute_side_gradient: side is not an edge of element");
    }
    double dxi[4], deta[4], dx[4], dy[4];
    q1::shape_derivatives(xi, eta, dxi, deta);
    q1::physical_derivatives(q1::jacobian(x, y, xi, eta), dxi, deta, dx, dy);
    for (int a = 0; a < 4; ++a) {
        const auto ip = static_cast<std::size_t>(nodes[a]);
        g.ux += u[ip] * dx[a];
        g.uy += u[ip] * dy[a];
        g.vx += v[ip] * dx[a];
        g.vy += v[ip] * dy[a];
    }
    return g;
}

} // namespace

std::vector<int> resolve_wall_sides(const Mesh& mesh, const std::string& patch_identifier) {
    if (patch_identifier.empty()) return {};

    // Check if numeric tag/flag
    int numeric_id = -1;
    try {
        std::size_t pos = 0;
        int val = std::stoi(patch_identifier, &pos);
        if (pos == patch_identifier.size()) numeric_id = val;
    } catch (...) {}

    std::vector<int> out;
    for (int ib = 0; ib < mesh.nboun; ++ib) {
        const BoundarySide& s = mesh.iside[static_cast<std::size_t>(ib)];
        if (numeric_id >= 0) {
            if (s.flag == numeric_id || s.side_id == numeric_id) {
                out.push_back(ib);
            }
        } else {
            // Match canonical name or aliases
            const auto res = mesh::resolve_boundary_name(patch_identifier);
            if (res.resolved()) {
                auto sf = mesh::solver_flag(res.type);
                if (sf.has_value() && s.flag == *sf) {
                    out.push_back(ib);
                }
            }
        }
    }
    return out;
}

SkinFrictionProfile extract_skin_friction(
    const Mesh& mesh,
    const std::vector<int>& boundary_sides,
    double nu,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& p,
    double u_ref,
    double p_ref,
    double rho) {
    if (boundary_sides.empty()) return {};

    const double q_inf = 0.5 * rho * u_ref * u_ref;
    const double inv_q = (q_inf > 0.0) ? (1.0 / q_inf) : 1.0;

    std::vector<SkinFrictionSample> raw_samples;
    raw_samples.reserve(boundary_sides.size());

    for (int ib : boundary_sides) {
        if (ib < 0 || ib >= mesh.nboun) continue;
        const BoundarySide& s = mesh.iside[static_cast<std::size_t>(ib)];
        const int n1 = s.node1;
        const int n2 = s.node2;
        const double x1 = mesh.x[static_cast<std::size_t>(n1)], y1 = mesh.y[static_cast<std::size_t>(n1)];
        const double x2 = mesh.x[static_cast<std::size_t>(n2)], y2 = mesh.y[static_cast<std::size_t>(n2)];

        const double dx = x2 - x1;
        const double dy = y2 - y1;
        const double length = std::hypot(dx, dy);
        if (length <= 0.0) continue;

        // Normal into fluid (towards element centroid)
        double nx = -dy / length;
        double ny = dx / length;
        const ElementNodes elem_nodes = mesh.element_nodes(s.elem);
        double cx = 0.0, cy = 0.0;
        for (int k = 0; k < elem_nodes.size(); ++k) {
            cx += mesh.x[static_cast<std::size_t>(elem_nodes[k])];
            cy += mesh.y[static_cast<std::size_t>(elem_nodes[k])];
        }
        cx /= elem_nodes.size();
        cy /= elem_nodes.size();

        if (nx * (cx - x1) + ny * (cy - y1) < 0.0) {
            nx = -nx;
            ny = -ny;
        }

        // Tangent vector: perpendicular to normal, oriented such that tx >= 0 (or ty >= 0 if vertical)
        double tx = -ny;
        double ty = nx;
        if (tx < -1e-12 || (std::abs(tx) <= 1e-12 && ty < 0.0)) {
            tx = -tx;
            ty = -ty;
        }

        // Pressure at midpoint
        const double p_mid = 0.5 * (p[static_cast<std::size_t>(n1)] + p[static_cast<std::size_t>(n2)]);
        const double cp = (p_mid - p_ref) * inv_q;

        // Velocity gradient at side midpoint
        const SideGradient g = compute_side_gradient(mesh, s.elem, n1, n2, u, v);

        // Viscous stress tensor: sigma_visc = mu * (grad u + grad u^T)
        // With rho, mu = rho * nu
        const double mu = rho * nu;
        const double sxx = 2.0 * g.ux;
        const double syy = 2.0 * g.vy;
        const double sxy = g.uy + g.vx;

        // Viscous traction vector: t_visc = sigma_visc . n_b
        const double trac_x = mu * (sxx * nx + sxy * ny);
        const double trac_y = mu * (sxy * nx + syy * ny);

        // Wall shear stress: tangential traction
        const double tau_w = trac_x * tx + trac_y * ty;
        const double cf = tau_w * inv_q;

        SkinFrictionSample sample;
        sample.side_index = ib;
        sample.node1 = n1;
        sample.node2 = n2;
        sample.x = 0.5 * (x1 + x2);
        sample.y = 0.5 * (y1 + y2);
        sample.length = length;
        sample.nx = nx;
        sample.ny = ny;
        sample.tx = tx;
        sample.ty = ty;
        sample.tau_w = tau_w;
        sample.cf = cf;
        sample.p = p_mid;
        sample.cp = cp;

        raw_samples.push_back(sample);
    }

    if (raw_samples.empty()) return {};

    // Sort samples geometrically along primary direction (increasing x, then y)
    std::sort(raw_samples.begin(), raw_samples.end(), [](const SkinFrictionSample& a, const SkinFrictionSample& b) {
        if (std::abs(a.x - b.x) > 1e-9) return a.x < b.x;
        return a.y < b.y;
    });

    // Compute cumulative arc length s along sorted samples
    double total_s = 0.0;
    for (const auto& smp : raw_samples) {
        total_s += smp.length;
    }
    raw_samples[0].s = 0.5 * raw_samples[0].length;
    for (std::size_t i = 1; i < raw_samples.size(); ++i) {
        const double d_mid = std::hypot(raw_samples[i].x - raw_samples[i - 1].x,
                                        raw_samples[i].y - raw_samples[i - 1].y);
        raw_samples[i].s = raw_samples[i - 1].s + d_mid;
    }

    SkinFrictionProfile profile;
    profile.samples = std::move(raw_samples);
    profile.u_ref = u_ref;
    profile.p_ref = p_ref;
    profile.rho = rho;
    profile.total_length = total_s;
    return profile;
}

std::vector<RecirculationBubble> find_recirculation_bubbles(const SkinFrictionProfile& profile) {
    const auto& s = profile.samples;
    if (s.size() < 2) return {};

    std::vector<RecirculationBubble> bubbles;
    bool in_bubble = false;
    RecirculationBubble current;

    // Check if initial point is already separated (e.g. backward facing step expansion)
    if (s[0].cf < 0.0) {
        in_bubble = true;
        current.x_sep = s[0].x - 0.5 * s[0].length * s[0].tx;
        current.y_sep = s[0].y - 0.5 * s[0].length * s[0].ty;
        current.s_sep = 0.0;
        current.cf_min = s[0].cf;
        current.x_cf_min = s[0].x;
    }

    for (std::size_t i = 0; i + 1 < s.size(); ++i) {
        const double cf1 = s[i].cf;
        const double cf2 = s[i + 1].cf;

        if (in_bubble) {
            // Track minimum Cf inside bubble
            if (cf1 < current.cf_min) {
                current.cf_min = cf1;
                current.x_cf_min = s[i].x;
            }
            if (cf1 < 0.0) {
                current.reverse_flux += -cf1 * s[i].length;
            }

            // Look for reattachment (Cf crossing from negative to non-negative)
            if (cf1 <= 0.0 && cf2 > 0.0) {
                const double alpha = (cf2 != cf1) ? (-cf1 / (cf2 - cf1)) : 0.5;
                current.s_reatt = s[i].s + alpha * (s[i + 1].s - s[i].s);
                current.x_reatt = s[i].x + alpha * (s[i + 1].x - s[i].x);
                current.y_reatt = s[i].y + alpha * (s[i + 1].y - s[i].y);
                current.length = current.s_reatt - current.s_sep;

                bubbles.push_back(current);
                in_bubble = false;
                current = RecirculationBubble{};
            }
        } else {
            // Look for separation (Cf crossing from positive to negative)
            if (cf1 >= 0.0 && cf2 < 0.0) {
                const double alpha = (cf2 != cf1) ? (cf1 / (cf1 - cf2)) : 0.5;
                current.s_sep = s[i].s + alpha * (s[i + 1].s - s[i].s);
                current.x_sep = s[i].x + alpha * (s[i + 1].x - s[i].x);
                current.y_sep = s[i].y + alpha * (s[i + 1].y - s[i].y);
                current.cf_min = cf2;
                current.x_cf_min = s[i + 1].x;
                in_bubble = true;
            }
        }
    }

    // If bubble extends to the end of the patch
    if (in_bubble) {
        const auto& last = s.back();
        current.s_reatt = last.s + 0.5 * last.length;
        current.x_reatt = last.x + 0.5 * last.length * last.tx;
        current.y_reatt = last.y + 0.5 * last.length * last.ty;
        current.length = current.s_reatt - current.s_sep;
        bubbles.push_back(current);
    }

    return bubbles;
}

std::string format_bubble_report(
    const std::vector<RecirculationBubble>& bubbles,
    const std::string& patch_name) {
    std::ostringstream oss;
    oss << "========================================================================\n"
        << " SEPARATION & RECIRCULATION BUBBLE DIAGNOSTICS: " << patch_name << "\n"
        << "========================================================================\n";
    if (bubbles.empty()) {
        oss << " No separation or recirculation bubbles detected (flow fully attached).\n";
        oss << "========================================================================\n";
        return oss.str();
    }

    oss << std::fixed << std::setprecision(5);
    for (std::size_t i = 0; i < bubbles.size(); ++i) {
        const auto& b = bubbles[i];
        oss << " Bubble #" << (i + 1) << ":\n"
            << "   Separation Point (x_sep, y_sep):     (" << b.x_sep << ", " << b.y_sep << ") m\n"
            << "   Reattachment Point (x_r, y_r):       (" << b.x_reatt << ", " << b.y_reatt << ") m\n"
            << "   Bubble Length (L_bubble):            " << b.length << " m\n"
            << "   Peak Negative Skin Friction (Cf_min):" << b.cf_min << " at x = " << b.x_cf_min << " m\n";
    }
    oss << "========================================================================\n";
    return oss.str();
}

} // namespace hydrama::post
