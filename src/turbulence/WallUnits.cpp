#include "turbulence/WallUnits.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace hydrama {

std::vector<double> wall_friction_velocity(const Mesh& mesh, const WallGeometry& walls,
                                           const std::vector<double>& u, const std::vector<double>& v,
                                           double nu) {
    std::vector<double> u_tau(walls.num_faces(), 0.0);
    for (int f = 0; f < walls.num_faces(); ++f) {
        const WallFace& face = walls.face(f);
        const auto& n = mesh.intma[face.elem];
        const double* dNdx = mesh.dNkdx[face.elem].dNdx;
        const double* dNdy = mesh.dNkdx[face.elem].dNdy;
        double dudx = 0.0, dudy = 0.0, dvdx = 0.0, dvdy = 0.0;
        for (int k = 0; k < Mesh::nep; ++k) {
            dudx += dNdx[k] * u[n[k]];
            dudy += dNdy[k] * u[n[k]];
            dvdx += dNdx[k] * v[n[k]];
            dvdy += dNdy[k] * v[n[k]];
        }
        // (G n) and its tangential component, t = (n_y, -n_x).
        const double gnx = dudx * face.nx + dudy * face.ny;
        const double gny = dvdx * face.nx + dvdy * face.ny;
        const double tau_w = nu * std::abs(face.ny * gnx - face.nx * gny);
        u_tau[f] = std::sqrt(tau_w);
    }
    return u_tau;
}

void yplus_field(const WallGeometry& walls, const std::vector<double>& u_tau_face, double nu,
                 std::vector<double>& yplus) {
    const std::vector<double>& d = walls.wall_distance();
    const std::vector<int>& nearest = walls.nearest_wall_face();
    yplus.assign(d.size(), 0.0);
    for (std::size_t i = 0; i < d.size(); ++i) {
        const int f = nearest[i];
        if (f >= 0) yplus[i] = d[i] * u_tau_face[f] / nu;
    }
}

WallUnitStats first_cell_yplus(const WallGeometry& walls, const std::vector<double>& u_tau_face, double nu) {
    WallUnitStats s;
    s.faces = walls.num_faces();
    if (s.faces == 0) return s;
    s.min = std::numeric_limits<double>::max();
    s.max = 0.0;
    double weighted = 0.0, length = 0.0;
    for (int f = 0; f < s.faces; ++f) {
        const WallFace& face = walls.face(f);
        const double yp = face.yp * u_tau_face[f] / nu;
        s.min = std::min(s.min, yp);
        s.max = std::max(s.max, yp);
        weighted += yp * face.length;
        length += face.length;
    }
    s.mean = weighted / length;
    return s;
}

} // namespace hydrama
