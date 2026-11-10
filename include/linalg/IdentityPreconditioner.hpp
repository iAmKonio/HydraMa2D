#pragma once

// M = I (no preconditioning): z = r.

#include "linalg/Preconditioner.hpp"

namespace hydrama::linalg {

class IdentityPreconditioner final : public Preconditioner {
public:
    void setup(const LinearOperator& a) override { n_ = a.size(); }
    void apply(ConstVectorView r, VectorView z) const override;
    std::size_t size() const override { return n_; }
    std::string_view name() const override { return "none"; }

private:
    std::size_t n_ = 0;
};

} // namespace hydrama::linalg
