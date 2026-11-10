#pragma once

// Turbulence model factory with an explicit table (no static self-
// registration: hydrama2d_core is a static library and unreferenced
// registrar objects would be dropped by the linker).

#include "turbulence/TurbulenceConfig.hpp"
#include "turbulence/TurbulenceModel.hpp"

#include <memory>

namespace hydrama {

// nullptr for model "none" (laminar). Throws std::invalid_argument for models
// that are not implemented yet or for unsupported option combinations.
std::unique_ptr<TurbulenceModel> make_turbulence_model(const TurbulenceConfig& config);

// True if make_turbulence_model() can build `model`.
bool is_turbulence_model_available(TurbulenceModelType model) noexcept;

} // namespace hydrama
