#include "linalg/Preconditioner.hpp"

namespace hydrama::linalg {

void Preconditioner::update(const LinearOperator& a, OperatorChange /*change*/) {
    setup(a);
}

} // namespace hydrama::linalg
