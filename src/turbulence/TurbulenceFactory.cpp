#include "turbulence/TurbulenceFactory.hpp"

#include "turbulence/KEpsilon.hpp"
#include "turbulence/KOmegaSST.hpp"
#include "turbulence/SpalartAllmaras.hpp"

#include <stdexcept>
#include <string>

namespace hydrama {

namespace {

using Creator = std::unique_ptr<TurbulenceModel> (*)(const TurbulenceConfig&);

std::unique_ptr<TurbulenceModel> create_spalart_allmaras(const TurbulenceConfig& config) {
    if (config.effective_wall_treatment() != WallTreatmentType::LowRe) {
        throw std::invalid_argument("spalart-allmaras supports wall_treatment low-re only (got " +
                                    std::string(to_string(config.effective_wall_treatment())) + ")");
    }
    return std::make_unique<SpalartAllmaras>(config);
}

void require_wall_treatment(const TurbulenceConfig& config, WallTreatmentType required, const std::string& hint) {
    if (config.effective_wall_treatment() != required) {
        throw std::invalid_argument(std::string(to_string(config.model)) + " supports wall_treatment " +
                                    std::string(to_string(required)) + " only (got " +
                                    std::string(to_string(config.effective_wall_treatment())) + ")" + hint);
    }
}

std::unique_ptr<TurbulenceModel> create_k_omega_sst(const TurbulenceConfig& config) {
    require_wall_treatment(config, WallTreatmentType::LowRe, "");
    return std::make_unique<KOmegaSST>(config);
}

std::unique_ptr<TurbulenceModel> create_k_epsilon(const TurbulenceConfig& config) {
    require_wall_treatment(config, WallTreatmentType::WallFunction,
                           "; use launder-sharma-k-epsilon for resolved walls");
    return std::make_unique<StandardKEpsilon>(config);
}

std::unique_ptr<TurbulenceModel> create_launder_sharma(const TurbulenceConfig& config) {
    require_wall_treatment(config, WallTreatmentType::LowRe, "; use k-epsilon for wall functions");
    return std::make_unique<LaunderSharmaKEpsilon>(config);
}

struct Entry {
    TurbulenceModelType model;
    Creator create;
};

// Adding a model = one line here.
constexpr Entry kRegistry[] = {
    {TurbulenceModelType::SpalartAllmaras, &create_spalart_allmaras},
    {TurbulenceModelType::KOmegaSST, &create_k_omega_sst},
    {TurbulenceModelType::KEpsilon, &create_k_epsilon},
    {TurbulenceModelType::LaunderSharmaKEpsilon, &create_launder_sharma},
};

} // namespace

bool is_turbulence_model_available(TurbulenceModelType model) noexcept {
    if (model == TurbulenceModelType::None) return true;
    for (const Entry& e : kRegistry)
        if (e.model == model) return true;
    return false;
}

std::unique_ptr<TurbulenceModel> make_turbulence_model(const TurbulenceConfig& config) {
    if (config.model == TurbulenceModelType::None) return nullptr;
    for (const Entry& e : kRegistry)
        if (e.model == config.model) return e.create(config);
    throw std::invalid_argument("turbulence model '" + std::string(to_string(config.model)) +
                                "' is not implemented yet");
}

} // namespace hydrama
