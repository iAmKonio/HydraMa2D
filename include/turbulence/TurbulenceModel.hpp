#pragma once

// Turbulence model interface (Strategy). The solver owns one model (or none
// for laminar runs) and calls advance() once per time step after CBS Step 3,
// with the new velocity. Virtual dispatch happens once per step; the element
// and node loops live in the concrete models (see TransportModelBase).

#include "Mesh.hpp"
#include "geometry/WallGeometry.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace hydrama {

// Velocity field the turbulence equations are advected with.
struct FlowView {
    const std::vector<double>& u;
    const std::vector<double>& v;
};

// Everything a model needs from the solver for one step. The mesh must have
// its geometry computed (detJ, shape function derivatives).
struct TurbulenceContext {
    const Mesh& mesh;
    const WallGeometry& walls;
    double nu;                         // molecular kinematic viscosity (1/Re)
    const std::vector<double>& delte;  // element time steps
    const std::vector<double>& elcoe2; // nodal 1 / sum_e(M_e,i / dt_e), from update_lhs()
};

// Named nodal field for output / restart.
struct FieldRef {
    std::string name;
    std::vector<double>* values = nullptr;
};

struct EquationResidual {
    std::string name;
    double relative_change = 0.0; // ||phi^{n+1} - phi^n||_2 / ||phi^{n+1}||_2
};

class TurbulenceModel {
public:
    virtual ~TurbulenceModel() = default;

    virtual std::string_view name() const = 0;

    // Allocate, set initial (free-stream) values and boundary conditions.
    virtual void initialize(const TurbulenceContext& ctx, const FlowView& flow) = 0;

    // Advance the turbulence equations by one (pseudo-)time step.
    virtual void advance(const TurbulenceContext& ctx, const FlowView& flow) = 0;

    // Eddy viscosity: nodal and element mean (used by Step 1).
    virtual const std::vector<double>& nut() const = 0;
    virtual const std::vector<double>& nut_elem() const = 0;

    // Largest diffusivity of the transported equations per element (for the
    // diffusive time step limit).
    virtual const std::vector<double>& max_diffusivity_elem() const = 0;

    // Output fields: transported variables and nu_t (VTK).
    virtual std::vector<FieldRef> fields() = 0;

    // Model state: the transported variables (restart files). After writing
    // into these vectors call refresh() to recompute the derived fields.
    virtual std::vector<FieldRef> state_fields() = 0;
    virtual void refresh(const TurbulenceContext& ctx) = 0;

    virtual std::vector<EquationResidual> residuals() const = 0;
};

} // namespace hydrama
