#pragma once

// Standard log-law wall functions (high-Re k-epsilon).
//
//   u+ = y+                       y+ <= y+_lam   (viscous sub-layer)
//   u+ = ln(E y+) / kappa         y+ >  y+_lam   (log layer), kappa = 0.41, E = 9.8
//
// y+_lam is the intersection of both laws (11.53 for the default constants).
// With the equilibrium friction velocity u_tau = C_mu^(1/4) sqrt(k_p) the
// wall shear stress is imposed through the eddy viscosity of the elements
// adjacent to the wall (decision D3: no-slip velocity kept):
//   nu_t,w = nu (y+ kappa / ln(E y+) - 1)   for y+ > y+_lam, 0 otherwise,
// so that (nu + nu_t,w) U_p / y_p = u_tau kappa U_p / ln(E y+) = tau_w (log law).
// The dissipation at the wall nodes follows from local equilibrium:
//   eps_w = C_mu^(3/4) k^(3/2) / (kappa y_p),
// and the production of k there is P_w = tau_w u_tau / (kappa y_p).
//
// References: B. E. Launder, D. B. Spalding, Comput. Methods Appl. Mech.
// Eng. 3 (1974) 269-289; OpenFOAM nutkWallFunction / epsilonWallFunction
// (same formulation, cell-based).

namespace hydrama::wallfn {

struct LogLaw {
    double kappa = 0.41;
    double E = 9.8;
};

// Intersection of u+ = y+ and u+ = ln(E y+)/kappa (fixed-point iteration).
double y_plus_laminar(const LogLaw& law = {});

// u+(y+) of the two-layer law.
double u_plus(double y_plus, const LogLaw& law = {});

// Wall eddy viscosity nu (y+ kappa / ln(E y+) - 1) for y+ > y_plus_lam, else 0.
double nut_wall(double y_plus, double nu, double y_plus_lam, const LogLaw& law = {});

// Equilibrium friction velocity C_mu^(1/4) sqrt(k).
double friction_velocity(double k, double c_mu);

// Equilibrium dissipation C_mu^(3/4) k^(3/2) / (kappa y_p).
double epsilon_wall(double k, double y_p, double c_mu, const LogLaw& law = {});

} // namespace hydrama::wallfn
