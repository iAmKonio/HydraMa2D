#include "mesh/MeshQuality.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace hydrama::mesh {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegree = 180.0 / kPi;

// Corner angles (degrees), sines and edge lengths of a counter-clockwise polygon.
template <int N>
ElementQuality polygon_quality(const double x[N], const double y[N], double ideal_angle, double jacobian_scale) {
    ElementQuality q;
    double lmin = std::numeric_limits<double>::infinity(), lmax = 0.0;
    double amin = std::numeric_limits<double>::infinity(), amax = -std::numeric_limits<double>::infinity();
    double smin = std::numeric_limits<double>::infinity();
    for (int k = 0; k < N; ++k) {
        const int next = (k + 1) % N, prev = (k + N - 1) % N;
        const double ax = x[next] - x[k], ay = y[next] - y[k];
        const double bx = x[prev] - x[k], by = y[prev] - y[k];
        const double la = std::hypot(ax, ay), lb = std::hypot(bx, by);
        lmin = std::min(lmin, la);
        lmax = std::max(lmax, la);
        const double cross = ax * by - ay * bx, dot = ax * bx + ay * by;
        double angle = std::atan2(cross, dot) * kDegree; // (-180, 180]: negative = reflex corner
        if (angle < 0.0) angle += 360.0;
        amin = std::min(amin, angle);
        amax = std::max(amax, angle);
        const double sine = (la > 0.0 && lb > 0.0) ? cross / (la * lb) : 0.0;
        smin = std::min(smin, sine);
    }
    q.min_angle = amin;
    q.max_angle = amax;
    q.aspect_ratio = lmin > 0.0 ? lmax / lmin : std::numeric_limits<double>::infinity();
    q.skewness = std::min(1.0, std::max((amax - ideal_angle) / (180.0 - ideal_angle), (ideal_angle - amin) / ideal_angle));
    q.scaled_jacobian = smin * jacobian_scale;
    return q;
}

void add(MetricSummary& s, double value, int element, bool larger_is_worse, int count_before) {
    if (count_before == 0) {
        s.min = s.max = value;
        s.mean = 0.0;
        s.worst_element = element;
    } else {
        const double worst = larger_is_worse ? s.max : s.min;
        if (larger_is_worse ? value > worst : value < worst) s.worst_element = element;
        s.min = std::min(s.min, value);
        s.max = std::max(s.max, value);
    }
    s.mean += value; // divided by the count at the end
}

void add(ElementTypeQuality& t, const ElementQuality& q, int element) {
    add(t.min_angle, q.min_angle, element, false, t.count);
    add(t.max_angle, q.max_angle, element, true, t.count);
    add(t.aspect_ratio, q.aspect_ratio, element, true, t.count);
    add(t.skewness, q.skewness, element, true, t.count);
    add(t.scaled_jacobian, q.scaled_jacobian, element, false, t.count);
    ++t.count;
}

void finish(ElementTypeQuality& t) {
    if (t.count == 0) return;
    for (MetricSummary* s : {&t.min_angle, &t.max_angle, &t.aspect_ratio, &t.skewness, &t.scaled_jacobian})
        s->mean /= t.count;
}

} // namespace

ElementQuality triangle_quality(const double x[3], const double y[3]) {
    return polygon_quality<3>(x, y, 60.0, 2.0 / std::sqrt(3.0));
}

ElementQuality quad_quality(const double x[4], const double y[4]) {
    return polygon_quality<4>(x, y, 90.0, 1.0);
}

ElementQuality element_quality(const Mesh& mesh, int e) {
    const ElementNodes nodes = mesh.element_nodes(e);
    double x[4], y[4];
    for (int k = 0; k < nodes.size(); ++k) {
        x[k] = mesh.x[static_cast<std::size_t>(nodes[k])];
        y[k] = mesh.y[static_cast<std::size_t>(nodes[k])];
    }
    return nodes.size() == 4 ? quad_quality(x, y) : triangle_quality(x, y);
}

SkewnessClass skewness_class(double skewness) {
    if (skewness < 0.25) return SkewnessClass::Excellent;
    if (skewness < 0.5) return SkewnessClass::Good;
    if (skewness < 0.75) return SkewnessClass::Fair;
    if (skewness < 0.9) return SkewnessClass::Poor;
    return SkewnessClass::Bad;
}

const char* to_string(SkewnessClass c) {
    switch (c) {
    case SkewnessClass::Excellent: return "excellent";
    case SkewnessClass::Good: return "good";
    case SkewnessClass::Fair: return "fair";
    case SkewnessClass::Poor: return "poor";
    case SkewnessClass::Bad: return "bad";
    }
    return "unknown";
}

MeshQualityReport assess_mesh_quality(const Mesh& mesh) {
    MeshQualityReport r;
    for (int e = 0; e < mesh.nelem; ++e) {
        const ElementQuality q = element_quality(mesh, e);
        add(mesh.is_quad(e) ? r.quads : r.triangles, q, e);
        ++r.skewness_histogram[static_cast<std::size_t>(skewness_class(q.skewness))];
        if (r.worst_element < 0 || q.skewness > r.worst_skewness) {
            r.worst_element = e;
            r.worst_skewness = q.skewness;
        }
    }
    finish(r.triangles);
    finish(r.quads);
    return r;
}

std::string format_quality_report(const Mesh& mesh, const MeshQualityReport& r) {
    std::ostringstream os;
    os << std::fixed;
    auto centre = [&](int e) {
        const ElementNodes nodes = mesh.element_nodes(e);
        double cx = 0.0, cy = 0.0;
        for (int k = 0; k < nodes.size(); ++k) {
            cx += mesh.x[static_cast<std::size_t>(nodes[k])] / nodes.size();
            cy += mesh.y[static_cast<std::size_t>(nodes[k])] / nodes.size();
        }
        std::ostringstream c;
        c << "(" << std::setprecision(6) << std::defaultfloat << cx << ", " << cy << ")";
        return c.str();
    };
    auto type_line = [&](const char* name, const ElementTypeQuality& t) {
        if (t.count == 0) return;
        os << "Mesh quality: " << t.count << " " << name << ": angles " << std::setprecision(1) << t.min_angle.min
           << " - " << t.max_angle.max << " deg, aspect ratio max " << std::setprecision(2) << t.aspect_ratio.max
           << " (mean " << t.aspect_ratio.mean << "), skewness max " << t.skewness.max << " (mean "
           << t.skewness.mean << "), scaled Jacobian min " << t.scaled_jacobian.min << "\n";
    };
    type_line("triangles", r.triangles);
    type_line("quadrilaterals", r.quads);
    os << "Mesh quality: equiangle skewness";
    for (int c = 0; c < kSkewnessClasses; ++c)
        os << (c ? ", " : " ") << r.skewness_histogram[static_cast<std::size_t>(c)] << " "
           << to_string(static_cast<SkewnessClass>(c));
    os << "\n";
    if (r.worst_element >= 0) {
        os << "Mesh quality: most skewed element: " << (mesh.is_quad(r.worst_element) ? "quadrilateral" : "triangle")
           << " (element index " << r.worst_element << ", 0-based), skewness " << std::setprecision(3)
           << r.worst_skewness << ", at " << centre(r.worst_element) << "\n";
    }
    const int bad = r.skewness_histogram[static_cast<std::size_t>(SkewnessClass::Bad)];
    if (bad > 0) {
        os << "WARNING: " << bad << " element(s) with equiangle skewness >= 0.9: accuracy and the time step may "
           << "suffer; improve the mesh there\n";
    }
    return os.str();
}

} // namespace hydrama::mesh
