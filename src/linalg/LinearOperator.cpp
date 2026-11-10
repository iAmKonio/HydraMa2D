#include "linalg/LinearOperator.hpp"

#include <stdexcept>
#include <string>

namespace hydrama::linalg {

void LinearOperator::diagonal(VectorView /*d*/) const {
    throw std::logic_error("linalg: operator '" + std::string(name()) + "' does not provide its diagonal");
}

} // namespace hydrama::linalg
