#pragma once

#include <array>
#include <string>
#include <vector>

namespace hydrama {

struct PeriodicPairConfig {
    std::string slave;
    std::string master;
    std::array<double, 2> translation = {0.0, 0.0};
    bool auto_translation = true;
};

struct PeriodicConfig {
    bool enabled = false;
    std::array<double, 2> driving_force = {0.0, 0.0}; // (fx, fy) body force driving channel/cascade
    std::vector<PeriodicPairConfig> pairs;

    bool has_driving_force() const noexcept {
        return driving_force[0] != 0.0 || driving_force[1] != 0.0;
    }
};

} // namespace hydrama
