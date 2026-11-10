#pragma once

// Point-implicit (linearised) nodal source update for transported scalars.
//
// The source per unit volume is split as  Q(phi) = P - D * phi  with D >= 0
// (Patankar 1980, "source-term linearisation"): production-type terms are
// explicit, destruction-type terms implicit. With the lumped mass M_i and
// the explicit transport right-hand side rhs_i (see ScalarTransport.hpp):
//
//   phi_i^{n+1} = (phi_i^n + elcoe2_i rhs_i + dt_i P_i) / (1 + dt_i D_i),
//   dt_i = M_i elcoe2_i   (= dt for a uniform time step; nodal effective
//                          time step with local time stepping).
//
// The destruction term can never change the sign of phi, whatever the time
// step (stiff near-wall destruction is handled at explicit cost).
//
// With enforce_positivity, the explicit increment E = elcoe2_i rhs_i + dt_i P_i
// is also weighted Patankar-style when it is negative (modified Patankar-
// Euler, Burchard, Deleersnijder & Meister 2003):
//
//   E >= 0:            phi^{n+1} = (phi^n + E) / (1 + dt D)
//   E <  0, phi^n > 0: phi^{n+1} = phi^n / (1 + dt D - E / phi^n)
//   E <  0, phi^n <= 0: E is dropped
//
// which guarantees phi^{n+1} >= 0 whenever phi^n >= 0, for any time step.
// It is used for k, epsilon and omega; it is NOT used for the SA variable
// (SA-neg allows negative values by design).
//
// Quadratic (Riccati) sinks: for sources Q = P - D phi - B phi^2 with B > 0
// (e.g. -C2 eps^2 / k) the update solves the backward-Euler equation
//   dt B phi^2 + (1 + dt D) phi - c0 = 0,   c0 = phi^n + E   (E as above)
// exactly for its non-negative root
//   phi = 2 c0 / (beta + sqrt(beta^2 + 4 dt B c0)),  beta = 1 + dt D   (beta >= 0)
//   phi = (-beta + sqrt(beta^2 + 4 dt B c0)) / (2 dt B)                (beta < 0),
// which is positive and bounded by the local equilibrium for any time step
// (no overshoot of an explicitly produced variable). B = 0 reproduces the
// linear update bit for bit. Quadratic sinks require enforce_positivity.
//
// References: S. V. Patankar, Numerical Heat Transfer and Fluid Flow (1980);
// H. Burchard, E. Deleersnijder, A. Meister, "A high-order conservative
// Patankar-type discretisation for stiff systems of production-destruction
// equations", Appl. Numer. Math. 47 (2003) 1-30.

#include <vector>

namespace hydrama {

// Accumulates one node's linearised source Q = P - D phi.
struct SourceSplit {
    double P = 0.0;
    double D = 0.0; // >= 0 unless B > 0
    double B = 0.0; // quadratic sink coefficient, >= 0

    // Term c * phi: decay (c < 0) is implicit, growth (c >= 0) explicit.
    void add_linear(double c, double phi) noexcept {
        if (c < 0.0) {
            D -= c;
        } else {
            P += c * phi;
        }
    }

    // Term with known value q: sources are explicit; sinks are made implicit
    // proportionally to phi (Patankar) when phi > 0, explicit otherwise.
    void add_value(double q, double phi) noexcept {
        if (q >= 0.0) {
            P += q;
        } else if (phi > 0.0) {
            D -= q / phi;
        } else {
            P += q;
        }
    }

    // Nonlinear term q(phi) with derivative dq/dphi, Newton-linearised about
    // the current phi (one backward-Euler Newton step): decaying terms
    // (dq/dphi < 0) are implicit, q(phi_new) ~ q + dq/dphi (phi_new - phi);
    // growing terms are explicit. Used for variables that may be negative
    // (e.g. ln omega), where P may become negative.
    void add_newton(double q, double dq_dphi, double phi) noexcept {
        if (dq_dphi < 0.0) {
            P += q - dq_dphi * phi;
            D -= dq_dphi;
        } else {
            P += q;
        }
    }

    // Term -b phi^2 (b >= 0), implicit (exact backward Euler with the rest).
    void add_quadratic_sink(double b) noexcept { B += b; }

    double value(double phi) const noexcept { return P - D * phi - B * phi * phi; }
};

// phi <- point-implicit update (all vectors nodal, size npoin).
void point_implicit_update(const std::vector<double>& nodal_mass,
                           const std::vector<double>& elcoe2,
                           const std::vector<double>& rhs,
                           const std::vector<double>& production,
                           const std::vector<double>& destruction,
                           bool enforce_positivity,
                           std::vector<double>& phi);

// Same with quadratic sink coefficients B (>= 0; requires enforce_positivity
// where B > 0). Nodes with B == 0 are updated exactly as by the overload above.
void point_implicit_update(const std::vector<double>& nodal_mass,
                           const std::vector<double>& elcoe2,
                           const std::vector<double>& rhs,
                           const std::vector<double>& production,
                           const std::vector<double>& destruction,
                           const std::vector<double>& quadratic,
                           bool enforce_positivity,
                           std::vector<double>& phi);

} // namespace hydrama
