#pragma once

// Non-owning decorator z = factor * M^{-1} r.
//
// Setup reuse for scaled operators: for a homogeneous preconditioner
// (PreconditionerTraits::homogeneous, M(c A) = c M(A)) that was set up for
// A_ref, the operator A = c A_ref is preconditioned exactly by
//
//     M(A)^{-1} = (1/c) M(A_ref)^{-1}  ->  ScaledPreconditioner(m, 1 / c)
//
// without repeating the (symbolic or numeric) setup. With factor == 1 the
// decorator is bit-for-bit transparent.
//
// The decorator is a view: setup()/update() throw std::logic_error (set up
// the wrapped preconditioner instead), and the wrapped object must outlive it.

#include "linalg/Preconditioner.hpp"

namespace hydrama::linalg {

class ScaledPreconditioner final : public Preconditioner {
public:
    // Throws std::invalid_argument unless factor is finite and > 0.
    ScaledPreconditioner(const Preconditioner& inner, double factor);

    void setup(const LinearOperator& a) override;
    void update(const LinearOperator& a, OperatorChange change) override;
    void apply(ConstVectorView r, VectorView z) const override;
    std::size_t size() const override { return inner_.size(); }
    PreconditionerTraits traits() const override { return inner_.traits(); }
    std::string_view name() const override { return inner_.name(); }

    double factor() const { return factor_; }
    const Preconditioner& inner() const { return inner_; }

private:
    const Preconditioner& inner_;
    double factor_;
};

} // namespace hydrama::linalg
