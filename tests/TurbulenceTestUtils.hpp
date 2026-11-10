#pragma once

// Shared helpers for driving turbulence models directly in unit tests:
// mesh + wall geometry + time-step data (Setup), grid builders, injectable
// wall-distance strategies and a homogeneous-decay driver.

#include "LinearSolver.hpp"
#include "MeshTestUtils.hpp"
#include "geometry/WallGeometry.hpp"
#include "turbulence/TurbulenceConfig.hpp"
#include "turbulence/TurbulenceModel.hpp"

#include <array>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace hydrama::test {

// Mesh (geometry computed) + wall geometry + time step arrays as the solver
// provides them (elcoe2 from update_lhs).
struct TurbSetup {
    Mesh mesh;
    WallGeometry walls;
    double nu = 1.0e-3;
    std::vector<double> delte, elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;

    TurbSetup(Mesh m, double nu_, double dt) : mesh(std::move(m)), nu(nu_) {
        compute_all_geometry(mesh);
        walls = WallGeometry(mesh);
        set_dt(dt);
    }
    TurbSetup(Mesh m, double nu_, double dt, const WallDistanceCalculator& distance) : mesh(std::move(m)), nu(nu_) {
        compute_all_geometry(mesh);
        walls = WallGeometry(mesh, distance);
        set_dt(dt);
    }
    void set_dt(double dt) {
        delte.assign(mesh.nelem, dt);
        update_lhs(mesh, delte, elcoe2, pdiag, gstif);
    }
    TurbulenceContext ctx() const { return {mesh, walls, nu, delte, elcoe2}; }
};

inline Mesh turb_grid(int nx, int ny, double lx, double ly, int flag = 500) {
    GridSpec g;
    g.lx = lx;
    g.ly = ly;
    g.nx = nx;
    g.ny = ny;
    return make_grid_mesh(g, flag);
}

// Channel: bottom/top walls, left inlet (503), right outlet (504).
inline Mesh turb_channel(int nx, int ny, double lx, double ly) {
    Mesh m = turb_grid(nx, ny, lx, ly);
    set_flags_by_side_id(m, {{1, 500}, {2, 504}, {3, 500}, {4, 503}});
    return m;
}

// Boundary override of `flag` with the given kind; set the optional values
// on the returned object.
inline TurbulenceBoundaryOverride boundary_override(int flag, BoundaryKind kind) {
    TurbulenceBoundaryOverride b;
    b.flag = flag;
    b.kind = kind;
    return b;
}

inline TurbulenceConfig model_config(TurbulenceModelType model) {
    TurbulenceConfig c;
    c.present = true;
    c.model = model;
    return c;
}

// Distance to a virtual wall at y = -y0 below the mesh.
class VirtualWallDistance final : public WallDistanceCalculator {
public:
    explicit VirtualWallDistance(double y0) : y0_(y0) {}
    std::string_view name() const override { return "virtual-wall"; }
    WallDistanceField compute(const Mesh& mesh, const std::vector<WallFace>&) const override {
        WallDistanceField f;
        f.nearest_face.assign(mesh.npoin, -1);
        f.distance.resize(mesh.npoin);
        for (int ip = 0; ip < mesh.npoin; ++ip) f.distance[ip] = mesh.y[ip] + y0_;
        return f;
    }

private:
    double y0_;
};

// The same wall distance at every node.
class ConstantDistance final : public WallDistanceCalculator {
public:
    explicit ConstantDistance(double d) : d_(d) {}
    std::string_view name() const override { return "constant"; }
    WallDistanceField compute(const Mesh& mesh, const std::vector<WallFace>&) const override {
        WallDistanceField f;
        f.nearest_face.assign(mesh.npoin, -1);
        f.distance.assign(mesh.npoin, d_);
        return f;
    }

private:
    double d_;
};

} // namespace hydrama::test
