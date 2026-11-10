#pragma once

// Permutations of dof numberings and symmetric permutation of CSR matrices.
//
// Convention: new_to_old[k] is the original index placed at position k of
// the new numbering. For the permutation matrix P with (P x)[k] =
// x[new_to_old[k]]:
//
//     gather:   x_new = P x          x_new[k] = x_old[new_to_old[k]]
//     scatter:  x_old = P^T x_new    x_old[new_to_old[k]] = x_new[k]
//     matrix:   B = P A P^T          B[k][l] = A[new_to_old[k]][new_to_old[l]]
//
// SymmetricPermutation is a reusable plan for B = P A P^T on a fixed
// pattern: the permuted pattern and the value map are computed once, after
// which permuting new values (e.g. every time step) is an O(nnz) gather.

#include "linalg/CsrMatrix.hpp"
#include "linalg/SparsityPattern.hpp"
#include "linalg/VectorView.hpp"

#include <memory>
#include <vector>

namespace hydrama::linalg {

class Permutation {
public:
    Permutation() = default; // size 0

    // Throws std::invalid_argument unless new_to_old is a permutation of
    // 0 .. n-1.
    explicit Permutation(std::vector<int> new_to_old);

    static Permutation identity(int n);

    int size() const { return static_cast<int>(new_to_old_.size()); }
    const std::vector<int>& new_to_old() const { return new_to_old_; }
    const std::vector<int>& old_to_new() const { return old_to_new_; }
    bool is_identity() const;

    Permutation inverse() const;

    // x_new = P x_old. Sizes must equal size(); x_new and x_old must not alias.
    void gather(ConstVectorView x_old, VectorView x_new) const;
    // x_old = P^T x_new. Sizes must equal size(); must not alias.
    void scatter(ConstVectorView x_new, VectorView x_old) const;

private:
    std::vector<int> new_to_old_;
    std::vector<int> old_to_new_;
};

class SymmetricPermutation {
public:
    // Plan for B = P A P^T for matrices on `pattern` (square). The source
    // pattern is kept (shared) so that matches() can be checked cheaply.
    SymmetricPermutation(std::shared_ptr<const SparsityPattern> pattern, Permutation permutation);

    const Permutation& permutation() const { return permutation_; }
    const std::shared_ptr<const SparsityPattern>& source_pattern() const { return source_; }
    const std::shared_ptr<const SparsityPattern>& permuted_pattern() const { return permuted_; }

    // True if matrices with this pattern can be permuted with the plan
    // (pointer identity or structural equality).
    bool matches(const std::shared_ptr<const SparsityPattern>& pattern) const;

    // b := P a P^T. a must match the plan; b's storage is reused when it
    // already has the permuted pattern. Symmetry is carried over.
    void apply(const CsrMatrix& a, CsrMatrix& b) const;
    CsrMatrix apply(const CsrMatrix& a) const;

private:
    std::shared_ptr<const SparsityPattern> source_;
    std::shared_ptr<const SparsityPattern> permuted_;
    Permutation permutation_;
    std::vector<int> value_map_; // b.values[k] = a.values[value_map_[k]]
};

} // namespace hydrama::linalg
