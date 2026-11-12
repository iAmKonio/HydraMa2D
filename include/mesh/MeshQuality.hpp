#pragma once

// Element quality metrics of triangles and quadrilaterals, and a summary for
// the run log (docs/MESH_IMPORT.md, section "Mesh quality").
//
// Per element, from the corner angles theta_k and the edge lengths L_k:
//
//   min_angle, max_angle   interior angles in degrees;
//   aspect_ratio           edge ratio L_max / L_min (>= 1; the "edge ratio" of
//                          the Verdict library, Stimpson et al. 2007). Large
//                          values are intended in boundary (inflation) layers;
//   skewness               equiangle skewness
//                              max((theta_max - theta_e) / (180 - theta_e),
//                                  (theta_e - theta_min) / theta_e),
//                          theta_e = 60 (triangle) or 90 (quadrilateral):
//                          0 for equilateral triangles and rectangles, 1 for
//                          degenerate elements (the definition of Gambit /
//                          ANSYS Fluent / Cubit);
//   scaled_jacobian        min over the corners of sin(theta_k), normalised
//                          by sin(60 deg) for triangles (Verdict): 1 for
//                          equilateral triangles and rectangles, <= 0 for
//                          inverted corners.
//
// Skewness classes (the usual guidelines): excellent < 0.25 <= good < 0.5
// <= fair < 0.75 <= poor < 0.9 <= bad.

#include "Mesh.hpp"

#include <array>
#include <string>

namespace hydrama::mesh {

struct ElementQuality {
    double min_angle = 0.0;       // degrees
    double max_angle = 0.0;       // degrees
    double aspect_ratio = 0.0;    // longest / shortest edge
    double skewness = 0.0;        // equiangle skewness, 0 (ideal) .. 1 (degenerate)
    double scaled_jacobian = 0.0; // 1 (ideal) .. <= 0 (inverted)
};

// Quality of element e (triangle or quadrilateral, counter-clockwise).
ElementQuality element_quality(const Mesh& mesh, int e);

// Quality of a triangle / quadrilateral given by its corner coordinates
// (counter-clockwise).
ElementQuality triangle_quality(const double x[3], const double y[3]);
ElementQuality quad_quality(const double x[4], const double y[4]);

// Minimum, mean and maximum of one metric over the elements of one type,
// with the element that has the worst value.
struct MetricSummary {
    double min = 0.0;
    double mean = 0.0;
    double max = 0.0;
    int worst_element = -1;
};

struct ElementTypeQuality {
    int count = 0;
    MetricSummary min_angle;       // worst: smallest
    MetricSummary max_angle;       // worst: largest
    MetricSummary aspect_ratio;    // worst: largest
    MetricSummary skewness;        // worst: largest
    MetricSummary scaled_jacobian; // worst: smallest
};

enum class SkewnessClass { Excellent, Good, Fair, Poor, Bad };
inline constexpr int kSkewnessClasses = 5;
SkewnessClass skewness_class(double skewness);
const char* to_string(SkewnessClass c);

struct MeshQualityReport {
    ElementTypeQuality triangles;
    ElementTypeQuality quads;
    std::array<int, kSkewnessClasses> skewness_histogram{}; // elements per SkewnessClass
    int worst_element = -1;     // largest skewness over all elements
    double worst_skewness = 0.0;
};

MeshQualityReport assess_mesh_quality(const Mesh& mesh);

// Run-log lines ("Mesh quality: ..."), with a warning line if elements have
// skewness >= 0.9 (class bad).
std::string format_quality_report(const Mesh& mesh, const MeshQualityReport& report);

} // namespace hydrama::mesh
