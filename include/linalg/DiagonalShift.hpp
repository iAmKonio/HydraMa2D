#pragma once

// Diagonal shift strategy for incomplete factorisations.
//
// An incomplete factorisation of A can break down (non-positive pivot in
// IC(0), zero pivot in ILU(0)) even for symmetric positive definite A unless
// A is an M-matrix (Meijerink & van der Vorst 1977; Kershaw 1978). The
// classical remedy (Manteuffel 1980) factors the shifted matrix
//
//     A + alpha diag(A)
//
// instead, increasing alpha until the factorisation succeeds. The schedule
// follows Lin & More (1999): alpha_0 = initial, then
// alpha_{k+1} = max(2 alpha_k, minimum).

namespace hydrama::linalg {

struct DiagonalShift {
    double initial = 0.0;    // alpha_0 (0: try the unshifted matrix first)
    double minimum = 1.0e-3; // first non-zero shift after a breakdown
    int max_attempts = 40;   // alpha reaches ~ 5e8 * minimum

    // A pivot p breaks down if it is not finite or |p| <= tolerance * |a_ii|
    // (p <= tolerance * a_ii for Cholesky).
    double pivot_tolerance = 1.0e-12;
};

} // namespace hydrama::linalg
