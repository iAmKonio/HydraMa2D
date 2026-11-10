# Linear Solver Layer — References

Sources behind the algorithms in `include/linalg/` and `src/linalg/`, with what each one was used for and where the implementation follows or departs from it. The header comment of every class cites its sources.

> **Note on accuracy.** This list was compiled without internet access. Section/algorithm numbers refer to the editions given, and DOIs are from memory: please verify them before citing in a publication.

---

## Sparse storage and assembly

| Reference | Used for |
| :--- | :--- |
| **Y. Saad**, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003. | CSR format (Sec. 3.4); finite element assembly into a global sparse matrix (Sec. 2.3); permutations and symmetric reordering `P A P^T` (Sec. 3.3). → `SparsityPattern`, `CsrMatrix`, `P1CsrAssembler`, `Permutation`, `SymmetricPermutation` |
| **R. Barrett, M. Berry, T. F. Chan, J. Demmel, J. Donato, J. Dongarra, V. Eijkhout, R. Pozo, C. Romine, H. van der Vorst**, *Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods*, 2nd ed., SIAM, 1994. | Compressed row storage and its matrix-vector product (Sec. 4.3); overview of Jacobi/SSOR/incomplete factorisation preconditioners (Sec. 3.4). |
| **O. C. Zienkiewicz, R. L. Taylor, J. Z. Zhu**, *The Finite Element Method: Its Basis and Fundamentals*, 7th ed., Butterworth-Heinemann, 2013. | Standard element-by-element assembly (precomputed local-to-global scatter). → `P1CsrAssembler` |

**Implementation notes.** The element scatter map (`map[e][3a+b]` = position of entry `(intma[e][a], intma[e][b])` in the CSR value array) is computed once per mesh, so re-assembly with new time steps takes O(nelem) with no searching. `assemble_ebe()` sums the same element contributions in the same order as the matrix-free operator, so the assembled entries are bit-identical to the EBE operator (tested).

---

## Reverse Cuthill–McKee ordering

| Reference | Used for |
| :--- | :--- |
| **E. Cuthill, J. McKee**, "Reducing the bandwidth of sparse symmetric matrices", *Proc. 24th National Conference of the ACM*, pp. 157–172, 1969. doi:10.1145/800195.805928 | The Cuthill–McKee algorithm: breadth-first numbering, visiting unnumbered neighbours in order of increasing degree. → `cuthill_mckee()` |
| **J. A. George**, *Computer Implementation of the Finite Element Method*, PhD thesis, Stanford University, Report STAN-CS-71-208, 1971. | Reversing the Cuthill–McKee order (RCM). → `reverse_cuthill_mckee()` |
| **W.-H. Liu, A. H. Sherman**, "Comparative analysis of the Cuthill–McKee and the reverse Cuthill–McKee ordering algorithms for sparse matrices", *SIAM J. Numer. Anal.* 13(2):198–213, 1976. doi:10.1137/0713020 | Result that the RCM envelope is never larger than the CM envelope (bandwidth equal). → property test `RCM envelope is never larger than the Cuthill-McKee envelope` |
| **A. George, J. W. H. Liu**, "An implementation of a pseudoperipheral node finder", *ACM Trans. Math. Software* 5(3):284–295, 1979. doi:10.1145/355841.355845 | Starting node: rooted level structures, restarting from a minimum-degree node of the last level while the eccentricity (number of levels) grows. → `pseudo_peripheral_node()` |
| **A. George, J. W. H. Liu**, *Computer Solution of Large Sparse Positive Definite Systems*, Prentice-Hall, 1981. | RCM per connected component, rooted level structures, envelope/profile definitions (Ch. 4). → `Reordering.cpp`, `SparsityPattern::profile()` |

**Implementation notes.** The graph is that of `A + A^T` without self-loops, so structurally non-symmetric patterns are handled. Components are processed in the order of their lowest-numbered node, and ties between equal degrees go to the lower original index. These are my own choices, made for determinism. On the Hinterkante mesh the bandwidth drops from 8376 to 126 and the lower envelope from 2.04·10⁷ to 6.3·10⁵.

---

## SSOR preconditioner

| Reference | Used for |
| :--- | :--- |
| **D. J. Evans**, "The use of pre-conditioning in iterative methods for solving linear equations with symmetric positive definite matrices", *IMA J. Appl. Math.* 4(3):295–314, 1968. doi:10.1093/imamat/4.3.295 | SSOR as a preconditioner for SPD systems. |
| **O. Axelsson**, *Iterative Solution Methods*, Cambridge University Press, 1994. | SSOR preconditioning matrix `M = (ω(2−ω))⁻¹ (D + ωL) D⁻¹ (D + ωU)` and its SPD property for SPD `A`, `0 < ω < 2` (Ch. 7). |
| **Y. Saad**, 2003 (above), Sec. 4.1 and 10.2. | Relaxation sweeps; SSOR as a preconditioner applied by one forward and one backward triangular sweep. |

**Implementation notes.** The factor `ω(2−ω)` is applied to the right-hand side of the forward sweep (linearity), so no extra pass is needed. `ω = 1` is symmetric Gauss–Seidel. Tests compare `M⁻¹` against the inverse of the closed form above for `ω ∈ {0.7, 1, 1.4}`.

---

## Incomplete factorisations IC(0) and ILU(0)

| Reference | Used for |
| :--- | :--- |
| **J. A. Meijerink, H. A. van der Vorst**, "An iterative solution method for linear systems of which the coefficient matrix is a symmetric M-matrix", *Math. Comp.* 31(137):148–162, 1977. doi:10.1090/S0025-5718-1977-0438681-4 | Incomplete Cholesky with the sparsity pattern of `A` (ICCG(0)) and its existence for symmetric M-matrices; incomplete LU. → `IncompleteCholesky`, `IncompleteLU` |
| **Y. Saad**, 2003 (above), Sec. 10.3, Algorithm 10.4 "ILU(0)" (IKJ variant). | Row-wise ILU(0) in CSR with a column-position work array; the property `(LU)ᵢⱼ = aᵢⱼ` on the pattern. → `IncompleteLU::factor()` and the pattern property tests for both factorisations |
| **G. H. Golub, C. F. Van Loan**, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013 (Cholesky, Ch. 4; incomplete factorisation preconditioners, Ch. 11). | Right-looking (KIJ) Cholesky, restricted to the pattern of `triu(A)`. → `IncompleteCholesky::factor()` |
| **D. S. Kershaw**, "The incomplete Cholesky—conjugate gradient method for the iterative solution of systems of linear equations", *J. Comput. Phys.* 26(1):43–65, 1978. doi:10.1016/0021-9991(78)90098-0 | IC(0) breakdown on SPD matrices that are not M-matrices; the 4×4 example used as a test (`kershaw_matrix()`, pivot −5 in the last row). |
| **T. A. Manteuffel**, "An incomplete factorization technique for positive definite linear systems", *Math. Comp.* 34(150):473–497, 1980. doi:10.1090/S0025-5718-1980-0559197-0 | Factorising the shifted matrix `A + α diag(A)` when the factorisation breaks down. → `DiagonalShift` |
| **C.-J. Lin, J. J. Moré**, "Incomplete Cholesky factorizations with limited memory", *SIAM J. Sci. Comput.* 21(1):24–45, 1999. doi:10.1137/S1064827597327334 | Shift schedule `α₀ = initial`, then `α_{k+1} = max(2α_k, α_S)` with `α_S = 10⁻³`. → `DiagonalShift` |

**Implementation notes.**
- IC(0) stores `U` (of `A ≈ UᵀU`) row-wise and reads only `triu(A)`. The forward solve with `Uᵀ` is column-oriented, the backward solve with `U` row-oriented.
- Lin & Moré scale `A` to unit diagonal and add `αI`. For a positive diagonal this is the same as Manteuffel's `A + α diag(A)`, which is what is implemented.
- The breakdown test (`pivot ≤ 10⁻¹² |aᵢᵢ|` or non-finite) and `max_attempts = 40` are my own choices.
- For SPD `A`, ILU(0) and IC(0) coincide in exact arithmetic (`U = D Lᵀ`). Both the unit tests (`M⁻¹` equal to 10⁻¹²) and the iteration counts on the benchmark mesh confirm this.

---

## Orderings for incomplete factorisations

| Reference | Used for |
| :--- | :--- |
| **I. S. Duff, G. A. Meurant**, "The effect of ordering on preconditioned conjugate gradients", *BIT* 29(4):635–657, 1989. doi:10.1007/BF01932738 | Motivation for applying RCM before IC(0)/SSOR in CG. |
| **M. Benzi, D. B. Szyld, A. van Duin**, "Orderings for incomplete factorization preconditioning of nonsymmetric problems", *SIAM J. Sci. Comput.* 20(5):1652–1670, 1999. doi:10.1137/S1064827597326845 | RCM as a robust ordering for ILU. → `ReorderedPreconditioner` |

**Measured effect (Hinterkante pressure matrix, CG to a relative residual of 1e-8).** The exact counts are pinned as orderings in `test_linalg_assembled_preconditioners.cpp`.

| Preconditioner | Iterations |
| :--- | :--- |
| Jacobi | 526 |
| SSOR | 248 |
| RCM+SSOR | 205 |
| IC(0) | 226 |
| RCM+IC(0) | 148 |
| ILU(0) | 226 |
| RCM+ILU(0) | 148 |

---

## Non-symmetric Krylov methods: BiCGSTAB and GMRES(m)

| Reference | Used for |
| :--- | :--- |
| **H. A. van der Vorst**, "Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems", *SIAM J. Sci. Stat. Comput.* 13(2):631–644, 1992. doi:10.1137/0913035 | The BiCGSTAB recurrences (ρ, α, ω, β). → `BiCGStab` |
| **R. Fletcher**, "Conjugate gradient methods for indefinite systems", in *Numerical Analysis Dundee 1975*, Lecture Notes in Mathematics 506, Springer, pp. 73–89, 1976. | BiCG, the basis of BiCGSTAB; the argument that BiCG (and so BiCGSTAB) terminates within the degree of the minimal polynomial. → test `BiCGStab terminates within the number of distinct eigenvalues` |
| **R. Barrett et al.**, *Templates*, 1994 (above), Sec. 2.3.8 and the preconditioned BiCGSTAB algorithm; Sec. 2.3.4 (GMRES). | Preconditioned BiCGSTAB with `p̂ = M⁻¹p`, `ŝ = M⁻¹s` (right preconditioning: the tracked residual is `b − A x`); the half-step convergence test on `s`; the breakdown conditions `ρ = 0` and `ω = 0`. |
| **G. L. G. Sleijpen, D. R. Fokkema**, "BiCGstab(ℓ) for linear equations involving unsymmetric matrices with complex spectrum", *ETNA* 1:11–32, 1993. | Stagnation of BiCGSTAB when `ω ≈ 0`; why `|(t,s)| ≈ 0` is detected and the half step kept. |
| **H. A. van der Vorst, Q. Ye**, "Residual replacement strategies for Krylov subspace iterative methods for the convergence of true residuals", *SIAM J. Sci. Comput.* 22(3):835–852, 2000. doi:10.1137/S1064827599353865 | The gap between recursively updated and true residuals; replacing the residual by `b − A x` (here: at claimed convergence) and restarting the recursion. → `BiCGStab::iterate()` |
| **Y. Saad, M. H. Schultz**, "GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems", *SIAM J. Sci. Stat. Comput.* 7(3):856–869, 1986. doi:10.1137/0907058 | GMRES and GMRES(m): Arnoldi basis, least-squares problem with the Hessenberg matrix, QR by Givens rotations updated column by column, residual norm `|g_{j+1}|` without forming the iterate, happy breakdown. → `Gmres` |
| **W. E. Arnoldi**, "The principle of minimized iterations in the solution of the matrix eigenvalue problem", *Quart. Appl. Math.* 9:17–29, 1951. | Arnoldi process. |
| **Y. Saad**, 2003 (above): Algorithm 6.9 (GMRES), Sec. 6.5.3 (practical implementation: progressive Givens rotations), Algorithm 7.7 (BiCGSTAB), Sec. 9.3 and Algorithm 9.5 (GMRES with right preconditioning). | Implementation structure of both methods; right vs. left preconditioning. |
| **G. H. Golub, C. F. Van Loan**, 2013 (above), Sec. 5.1 (Givens rotations). | Stable computation of `c, s` (here via `std::hypot`). |
| **C. C. Paige, M. Rozložník, Z. Strakoš**, "Modified Gram–Schmidt (MGS), least squares, and backward stability of MGS-GMRES", *SIAM J. Matrix Anal. Appl.* 28(1):264–284, 2006. doi:10.1137/050630416 | Justification of MGS (without reorthogonalisation) as the default Arnoldi orthogonalisation. |
| **L. Giraud, J. Langou, M. Rozložník, J. van den Eshof**, "Rounding error analysis of the classical Gram–Schmidt orthogonalization process", *Numer. Math.* 101:87–100, 2005; **B. N. Parlett**, *The Symmetric Eigenvalue Problem*, Prentice-Hall, 1980 (Kahan's "twice is enough"). | Classical Gram–Schmidt applied twice (CGS2). → `Orthogonalization::ClassicalGramSchmidtTwice` |
| **S. C. Eisenstat, H. C. Elman, M. H. Schultz**, "Variational iterative methods for nonsymmetric systems of linear equations", *SIAM J. Numer. Anal.* 20(2):345–357, 1983. doi:10.1137/0720023 (and H. C. Elman, PhD thesis, Yale, 1982) | Convergence of restarted minimal-residual methods, including GMRES(1), for positive-real matrices. → test `GMRES(1) converges for positive-real matrices` |
| **M. Embree**, "The tortoise and the hare restart GMRES", *SIAM Review* 45(2):259–266, 2003. | Larger restart lengths are not always faster. This is why the tests only compare full GMRES against restarted GMRES, not two restart lengths. |

**Implementation notes.**
- Both methods are right-preconditioned, so the convergence criteria refer to the unpreconditioned residual, as for CG.
- **BiCGSTAB** detects the pivot breakdown `|(r̃,v)| ≤ 10⁻¹⁴‖r̃‖‖v‖`, the stagnation `|(t,s)| ≤ 10⁻¹⁴‖t‖‖s‖` and the Lanczos breakdown `|(r̃,r)| ≤ 10⁻¹⁴‖r̃‖‖r‖`. The thresholds are my own choice. The three test systems were constructed symbolically (sympy) so that these quantities vanish exactly; in double precision they are ≤ 4·10⁻¹⁶ relative.
- **Verifying convergence.** When the recursive (BiCGSTAB) or estimated (GMRES) residual meets the tolerance, the true residual `b − A x` is computed and reported. If it misses the tolerance, the method restarts from the true residual.
- **Stagnation.** If the true residual fails to halve over two consecutive such false convergences, the solve ends with `StopReason::Stagnation`. This `StagnationGuard` is my own safeguard against looping at the attainable accuracy, motivated by van der Vorst & Ye.
- **GMRES storage.** GMRES stores only the `(m+1) × m` Hessenberg matrix (overwritten by `R`), `m+1` basis vectors and the rotations. The happy-breakdown threshold is `h_{j+1,j} ≤ 10⁻¹⁴‖A M⁻¹ v_j‖`.

---

## Setup reuse under uniform time stepping

No external reference: this is an elementary property, and the design is my own. For `A = dt K` with constant `K`, every preconditioner of the suite is homogeneous of degree one:

- Jacobi: `D`.
- SSOR: `(D + ωL) D⁻¹ (D + ωU)`.
- IC(0) and ILU(0): factors of `c A` are `√c U` and `L, c U`.

The IC(0)/ILU(0) breakdown tests and the shift schedule are invariant under scaling. Hence `M(c A)⁻¹ = c⁻¹ M(A)⁻¹`, so a factorisation computed once serves all later uniform time steps through `ScaledPreconditioner`.

`PressureSolver` does not trust the caller. It verifies `A = c A_ref` entrywise (relative 10⁻¹²) from the element data before reusing, so local time steps and geometry changes are detected automatically. On the Dirichlet-constrained rows, `M(A_ref)` and `M(A)` differ (unit rows are not scaled), but those rows never enter the iteration: residuals and corrections vanish there, and the solvers project preconditioned directions.

---

## Krylov method and convergence control (previous phase)

| Reference | Used for |
| :--- | :--- |
| **M. R. Hestenes, E. Stiefel**, "Methods of conjugate gradients for solving linear systems", *J. Res. Nat. Bur. Standards* 49(6):409–436, 1952. doi:10.6028/jres.049.044 | Conjugate gradient method. → `ConjugateGradient` |
| **Y. Saad**, 2003 (above), Algorithm 6.18 (CG) and Sec. 9.2 (preconditioned CG). | Preconditioned CG loop structure. |
| **R. Barrett et al.**, 1994 (above), Sec. 4.2 (stopping criteria). | Residual-based relative/absolute stopping tests. → `ConvergenceControl` |

**Implementation notes.** The legacy stopping rule (relative tolerance measured against the residual after the first iteration, zero-step breakdown guards) reproduces the original HydraMa Fortran/C++ PCG. It is kept bit-for-bit compatible; see `tests/reference/LegacyPcg.hpp`.
