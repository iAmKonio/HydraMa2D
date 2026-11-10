#include "turbulence/TurbulenceConfig.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace hydrama {

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

namespace {

std::string lower(std::string_view s) {
    std::string r(s);
    for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

struct ModelName {
    std::string_view name;
    TurbulenceModelType model;
};

// First entry per model is the canonical name.
constexpr ModelName kModelNames[] = {
    {"none", TurbulenceModelType::None},
    {"laminar", TurbulenceModelType::None},
    {"spalart-allmaras", TurbulenceModelType::SpalartAllmaras},
    {"sa", TurbulenceModelType::SpalartAllmaras},
    {"sa-neg", TurbulenceModelType::SpalartAllmaras},
    {"k-epsilon", TurbulenceModelType::KEpsilon},
    {"rng-k-epsilon", TurbulenceModelType::RngKEpsilon},
    {"realizable-k-epsilon", TurbulenceModelType::RealizableKEpsilon},
    {"launder-sharma-k-epsilon", TurbulenceModelType::LaunderSharmaKEpsilon},
    {"k-omega-sst", TurbulenceModelType::KOmegaSST},
    {"sst", TurbulenceModelType::KOmegaSST},
};

struct TreatmentName {
    std::string_view name;
    WallTreatmentType treatment;
};

constexpr TreatmentName kTreatmentNames[] = {
    {"low-re", WallTreatmentType::LowRe},
    {"wall-function", WallTreatmentType::WallFunction},
    {"automatic", WallTreatmentType::Automatic},
};

} // namespace

std::string_view to_string(TurbulenceModelType model) noexcept {
    for (const auto& e : kModelNames)
        if (e.model == model) return e.name;
    return "unknown";
}

std::string_view to_string(WallTreatmentType treatment) noexcept {
    for (const auto& e : kTreatmentNames)
        if (e.treatment == treatment) return e.name;
    return "unknown";
}

std::optional<TurbulenceModelType> parse_turbulence_model(std::string_view name) {
    const std::string key = lower(name);
    for (const auto& e : kModelNames)
        if (e.name == key) return e.model;
    return std::nullopt;
}

std::optional<WallTreatmentType> parse_wall_treatment(std::string_view name) {
    const std::string key = lower(name);
    for (const auto& e : kTreatmentNames)
        if (e.name == key) return e.treatment;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

const TurbulenceBoundaryOverride* TurbulenceConfig::find_boundary(int flag) const noexcept {
    for (const auto& b : boundaries)
        if (b.flag == flag) return &b;
    return nullptr;
}

WallTreatmentType TurbulenceConfig::effective_wall_treatment() const noexcept {
    if (wall_treatment) return *wall_treatment;
    switch (model) {
    case TurbulenceModelType::KEpsilon:
    case TurbulenceModelType::RngKEpsilon:
    case TurbulenceModelType::RealizableKEpsilon:
        return WallTreatmentType::WallFunction;
    default:
        return WallTreatmentType::LowRe;
    }
}

BoundaryKind TurbulenceConfig::boundary_kind(int flag) const noexcept {
    const TurbulenceBoundaryOverride* b = find_boundary(flag);
    return b ? b->kind : classify_boundary_flag(flag);
}

InletTurbulence TurbulenceConfig::inlet_for(int flag) const {
    InletTurbulence r = inlet;
    if (const TurbulenceBoundaryOverride* b = find_boundary(flag)) {
        if (b->intensity) r.intensity = *b->intensity;
        if (b->length_scale) r.length_scale = *b->length_scale;
        if (b->viscosity_ratio) r.viscosity_ratio = b->viscosity_ratio;
        if (b->nu_tilde_ratio) r.nu_tilde_ratio = b->nu_tilde_ratio;
        if (b->k) r.k = b->k;
        if (b->epsilon) r.epsilon = b->epsilon;
        if (b->omega) r.omega = b->omega;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

namespace {

class ParseError {
public:
    explicit ParseError(std::string msg) : msg_(std::move(msg)) {}
    const std::string& what() const { return msg_; }

private:
    std::string msg_;
};

std::vector<std::string> tokenize(const std::string& line) {
    const std::string content = line.substr(0, line.find('#'));
    std::istringstream ss(content);
    std::vector<std::string> tokens;
    std::string t;
    while (ss >> t) tokens.push_back(t);
    return tokens;
}

double parse_number(const std::string& token, const std::string& what) {
    std::string s = token;
    for (char& c : s)
        if (c == 'd' || c == 'D') c = 'e';
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE || !std::isfinite(v)) {
        throw ParseError("invalid number '" + token + "' for " + what);
    }
    return v;
}

double parse_positive(const std::string& token, const std::string& what) {
    const double v = parse_number(token, what);
    if (!(v > 0.0)) throw ParseError(what + " must be > 0, got " + token);
    return v;
}

double parse_non_negative(const std::string& token, const std::string& what) {
    const double v = parse_number(token, what);
    if (v < 0.0) throw ParseError(what + " must be >= 0, got " + token);
    return v;
}

int parse_int(const std::string& token, const std::string& what) {
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(token.c_str(), &end, 10);
    if (token.empty() || end != token.c_str() + token.size() || errno == ERANGE || v < -2147483647L ||
        v > 2147483647L) {
        throw ParseError("invalid integer '" + token + "' for " + what);
    }
    return static_cast<int>(v);
}

bool parse_bool(const std::string& token, const std::string& what) {
    const std::string t = lower(token);
    if (t == "on" || t == "true" || t == "yes" || t == "1") return true;
    if (t == "off" || t == "false" || t == "no" || t == "0") return false;
    throw ParseError("invalid switch '" + token + "' for " + what + " (use on/off)");
}

BoundaryKind parse_boundary_kind(const std::string& token) {
    const std::string t = lower(token);
    if (t == "wall") return BoundaryKind::Wall;
    if (t == "inlet") return BoundaryKind::Inlet;
    if (t == "outlet") return BoundaryKind::Outlet;
    if (t == "symmetry") return BoundaryKind::Symmetry;
    throw ParseError("unknown boundary kind '" + token + "' (use wall, inlet, outlet or symmetry)");
}

void expect_args(const std::vector<std::string>& tokens, std::size_t n) {
    if (tokens.size() != n + 1) {
        throw ParseError("'" + tokens[0] + "' expects " + std::to_string(n) + " value" + (n == 1 ? "" : "s") +
                         ", got " + std::to_string(tokens.size() - 1));
    }
}

TurbulenceBoundaryOverride parse_boundary(const std::vector<std::string>& tokens) {
    if (tokens.size() < 3) throw ParseError("'boundary' expects: boundary <flag> <kind> [key value]...");
    TurbulenceBoundaryOverride b;
    b.flag = parse_int(tokens[1], "boundary flag");
    b.kind = parse_boundary_kind(tokens[2]);
    if ((tokens.size() - 3) % 2 != 0) throw ParseError("boundary " + tokens[1] + ": options must be key value pairs");
    std::set<std::string> seen;
    for (std::size_t k = 3; k < tokens.size(); k += 2) {
        const std::string& key = tokens[k];
        const std::string& val = tokens[k + 1];
        if (!seen.insert(key).second) throw ParseError("boundary " + tokens[1] + ": duplicate option '" + key + "'");
        const std::string what = "boundary " + tokens[1] + " " + key;
        if (key == "intensity") {
            b.intensity = parse_non_negative(val, what);
        } else if (key == "length_scale") {
            b.length_scale = parse_positive(val, what);
        } else if (key == "viscosity_ratio") {
            b.viscosity_ratio = parse_non_negative(val, what);
        } else if (key == "nu_tilde_ratio") {
            b.nu_tilde_ratio = parse_non_negative(val, what);
        } else if (key == "k") {
            b.k = parse_non_negative(val, what);
        } else if (key == "epsilon") {
            b.epsilon = parse_positive(val, what);
        } else if (key == "omega") {
            b.omega = parse_positive(val, what);
        } else {
            throw ParseError("boundary " + tokens[1] + ": unknown option '" + key +
                             "' (intensity, length_scale, viscosity_ratio, nu_tilde_ratio, k, epsilon, omega)");
        }
    }
    if (b.kind != BoundaryKind::Inlet && !seen.empty()) {
        throw ParseError("boundary " + tokens[1] + ": inlet options given for a non-inlet boundary");
    }
    return b;
}

void parse_line(const std::vector<std::string>& tokens, TurbulenceConfig& c, std::set<std::string>& seen_keys) {
    const std::string& key = tokens[0];
    if (key == "boundary") {
        TurbulenceBoundaryOverride b = parse_boundary(tokens);
        if (c.find_boundary(b.flag)) throw ParseError("duplicate boundary entry for flag " + tokens[1]);
        c.boundaries.push_back(b);
        return;
    }
    if (!seen_keys.insert(key).second) throw ParseError("duplicate key '" + key + "'");

    if (key == "model") {
        expect_args(tokens, 1);
        const auto m = parse_turbulence_model(tokens[1]);
        if (!m) throw ParseError("unknown model '" + tokens[1] + "'");
        c.model = *m;
    } else if (key == "wall_treatment") {
        expect_args(tokens, 1);
        const auto t = parse_wall_treatment(tokens[1]);
        if (!t) throw ParseError("unknown wall_treatment '" + tokens[1] + "' (low-re, wall-function, automatic)");
        c.wall_treatment = *t;
    } else if (key == "wall_distance") {
        expect_args(tokens, 1);
        const auto m = parse_wall_distance_method(lower(tokens[1]));
        if (!m) throw ParseError("unknown wall_distance '" + tokens[1] + "' (exact, brute-force)");
        c.wall_distance = *m;
    } else if (key == "gradient_recovery") {
        expect_args(tokens, 1);
        const auto m = parse_gradient_recovery(lower(tokens[1]));
        if (!m) throw ParseError("unknown gradient_recovery '" + tokens[1] + "' (ppr, lumped-projection)");
        c.gradient_recovery = *m;
    } else if (key == "inlet_intensity") {
        expect_args(tokens, 1);
        c.inlet.intensity = parse_non_negative(tokens[1], key);
    } else if (key == "inlet_length_scale") {
        expect_args(tokens, 1);
        c.inlet.length_scale = parse_positive(tokens[1], key);
    } else if (key == "inlet_viscosity_ratio") {
        expect_args(tokens, 1);
        c.inlet.viscosity_ratio = parse_non_negative(tokens[1], key);
    } else if (key == "inlet_nu_tilde_ratio") {
        expect_args(tokens, 1);
        c.inlet.nu_tilde_ratio = parse_non_negative(tokens[1], key);
    } else if (key == "inlet_k") {
        expect_args(tokens, 1);
        c.inlet.k = parse_non_negative(tokens[1], key);
    } else if (key == "inlet_epsilon") {
        expect_args(tokens, 1);
        c.inlet.epsilon = parse_positive(tokens[1], key);
    } else if (key == "inlet_omega") {
        expect_args(tokens, 1);
        c.inlet.omega = parse_positive(tokens[1], key);
    } else if (key == "reference_velocity") {
        expect_args(tokens, 1);
        c.reference_velocity = parse_positive(tokens[1], key);
    } else if (key == "production_limiter") {
        expect_args(tokens, 1);
        c.production_limiter = parse_non_negative(tokens[1], key);
    } else if (key == "sa_ft2") {
        expect_args(tokens, 1);
        c.sa_ft2 = parse_bool(tokens[1], key);
    } else if (key == "nut_ratio_max") {
        expect_args(tokens, 1);
        c.nut_ratio_max = parse_positive(tokens[1], key);
    } else {
        throw ParseError("unknown key '" + key + "'");
    }
}

} // namespace

bool parse_turbulence_config(std::istream& in, TurbulenceConfig& config, std::string& error) {
    config = TurbulenceConfig{};
    config.present = true;
    std::set<std::string> seen_keys;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::vector<std::string> tokens = tokenize(line);
        if (tokens.empty()) continue;
        try {
            parse_line(tokens, config, seen_keys);
        } catch (const ParseError& e) {
            error = "line " + std::to_string(line_no) + ": " + e.what();
            return false;
        }
    }
    error.clear();
    return true;
}

bool read_turbulence_config(const std::string& filepath, TurbulenceConfig& config, std::string& error) {
    std::error_code ec;
    if (!std::filesystem::exists(filepath, ec)) {
        config = TurbulenceConfig{}; // laminar
        error.clear();
        return true;
    }
    std::ifstream in(filepath);
    if (!in.is_open()) {
        error = "cannot open " + filepath;
        return false;
    }
    if (!parse_turbulence_config(in, config, error)) {
        error = filepath + ": " + error;
        return false;
    }
    return true;
}

} // namespace hydrama
