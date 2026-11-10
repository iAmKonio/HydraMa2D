#pragma once

// Umbrella header of the HydraMa linear algebra layer (hydrama::linalg).
//
//   VectorView / VectorOps        non-owning vectors and deterministic kernels
//   LinearOperator                y = A x                     (Strategy)
//     EbeOperator                 matrix-free element-by-element operator
//     CsrMatrix                   assembled sparse matrix (SparsityPattern)
//     ConstrainedOperator         Dirichlet rows removed      (Decorator)
//   P1CsrAssembler                element-to-CSR assembly (pattern + scatter map)
//   Permutation / SymmetricPermutation, reverse_cuthill_mckee   (reordering)
//   DirichletConstraints          x[dof] = value
//   Preconditioner                z = M^{-1} r                (Strategy)
//     IdentityPreconditioner, JacobiPreconditioner
//     SsorPreconditioner, IncompleteCholesky (IC(0)), IncompleteLU (ILU(0))
//     ReorderedPreconditioner     M built in another numbering (Decorator)
//     ScaledPreconditioner        (1/c) M^{-1}: setup reuse for c A (Decorator)
//   KrylovSolver                  common solve() driver       (Template Method)
//     ConjugateGradient           SPD systems
//     BiCGStab, Gmres             general (non-symmetric) systems
//   make_preconditioner / make_krylov_solver                   (Factory)
//   ConvergenceCriteria / ConvergenceControl / SolveResult / StopReason
//   IterationMonitor              solve observers             (Observer)
//     ResidualHistory, StreamMonitor, MonitorGroup (Composite)

#include "linalg/BiCGStab.hpp"
#include "linalg/ConjugateGradient.hpp"
#include "linalg/ConstrainedOperator.hpp"
#include "linalg/Convergence.hpp"
#include "linalg/CsrAssembly.hpp"
#include "linalg/CsrMatrix.hpp"
#include "linalg/DiagonalShift.hpp"
#include "linalg/DirichletConstraints.hpp"
#include "linalg/EbeOperator.hpp"
#include "linalg/Factory.hpp"
#include "linalg/Gmres.hpp"
#include "linalg/IdentityPreconditioner.hpp"
#include "linalg/IncompleteCholesky.hpp"
#include "linalg/IncompleteLU.hpp"
#include "linalg/IterationMonitor.hpp"
#include "linalg/JacobiPreconditioner.hpp"
#include "linalg/KrylovSolver.hpp"
#include "linalg/LinearOperator.hpp"
#include "linalg/Permutation.hpp"
#include "linalg/Preconditioner.hpp"
#include "linalg/ReorderedPreconditioner.hpp"
#include "linalg/Reordering.hpp"
#include "linalg/ScaledPreconditioner.hpp"
#include "linalg/SparsityPattern.hpp"
#include "linalg/SsorPreconditioner.hpp"
#include "linalg/VectorOps.hpp"
#include "linalg/VectorView.hpp"
