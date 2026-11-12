#include "mesh/Su2Reader.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace hydrama::mesh {

namespace {

struct VtkTypeInfo {
    int type;
    int nodes;
    int dim;
    std::string_view name;
};

// VTK cell types used by SU2 (SU2 documentation, "Mesh File").
constexpr VtkTypeInfo kVtkTypes[] = {
    {1, 1, 0, "vertex"},       {3, 2, 1, "line"},         {5, 3, 2, "triangle"},  {9, 4, 2, "quadrilateral"},
    {10, 4, 3, "tetrahedron"}, {12, 8, 3, "hexahedron"},  {13, 6, 3, "prism"},    {14, 5, 3, "pyramid"},
};

const VtkTypeInfo* find_vtk_type(std::int64_t type) {
    for (const VtkTypeInfo& t : kVtkTypes)
        if (t.type == type) return &t;
    return nullptr;
}

bool is_blank(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }

std::string_view trim(std::string_view s) {
    while (!s.empty() && is_blank(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_blank(s.back())) s.remove_suffix(1);
    return s;
}

std::vector<std::string_view> split(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && is_blank(s[i])) ++i;
        const std::size_t start = i;
        while (i < s.size() && !is_blank(s[i])) ++i;
        if (i > start) out.push_back(s.substr(start, i - start));
    }
    return out;
}

bool parse_int(std::string_view s, std::int64_t& out) {
    if (!s.empty() && s.front() == '+') s.remove_prefix(1);
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc() && r.ptr == s.data() + s.size();
}

bool parse_double(std::string_view s, double& out) {
    if (!s.empty() && s.front() == '+') s.remove_prefix(1);
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc() && r.ptr == s.data() + s.size() && std::isfinite(out);
}

std::string excerpt(std::string_view s) {
    std::string out(s.substr(0, 60));
    for (char& c : out)
        if (static_cast<unsigned char>(c) < 32 || static_cast<unsigned char>(c) > 126) c = '?';
    if (s.size() > 60) out += "...";
    return out;
}

std::string upper(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

class Parser {
public:
    Parser(std::string_view buf, std::string source) : buf_(buf), source_(std::move(source)) {
        data_.source = source_;
        data_.format = "SU2 ASCII";
    }

    MeshData parse() {
        std::string_view line;
        while (next_line(line)) {
            std::string key;
            std::string_view value;
            if (!split_keyword(line, key, value)) {
                fail("unexpected data line '" + excerpt(line) +
                     "' outside a NELEM=, NPOIN= or NMARK= section (does the section count match its lines?)");
            }
            if (key.rfind("FFD_", 0) == 0) break; // free-form deformation boxes: always last, not mesh data
            if (key == "NDIME") {
                read_ndime(value);
            } else if (key == "NZONE") {
                const std::int64_t n = count(value, "NZONE=");
                if (n != 1) fail("multi-zone meshes (NZONE= " + std::to_string(n) + ") are not supported");
            } else if (key == "IZONE") {
                std::int64_t z = 0;
                if (!parse_int(value, z) || z != 1) fail("invalid IZONE= '" + excerpt(value) + "' (single-zone meshes only)");
            } else if (key == "NELEM") {
                require_once(have_elements_, "NELEM=");
                read_elements(value);
            } else if (key == "NPOIN") {
                require_once(have_points_, "NPOIN=");
                read_points(value);
            } else if (key == "NMARK") {
                require_once(have_markers_, "NMARK=");
                read_markers(value);
            } else if (key == "NPERIODIC") {
                require_once(have_periodic_, "NPERIODIC=");
                read_periodic(value);
            } else if (key == "MARKER_TAG" || key == "MARKER_ELEMS" || key == "PERIODIC_INDEX") {
                fail(key + "= outside its section (" + (key == "PERIODIC_INDEX" ? "NPERIODIC=" : "NMARK=") +
                     " count too small?)");
            } else {
                fail("unknown keyword '" + key + "='");
            }
        }
        if (ndime_ == 0) fail_at(0, "no NDIME= line: not an SU2 mesh file");
        if (!have_elements_) fail_at(0, "no NELEM= section");
        if (!have_points_) fail_at(0, "no NPOIN= section");
        if (!have_markers_) fail_at(0, "no NMARK= section (boundary markers)");
        check_node_indices();
        return std::move(data_);
    }

private:
    std::string_view buf_;
    std::size_t pos_ = 0;
    int line_no_ = 0;
    std::string source_;
    MeshData data_;

    int ndime_ = 0;
    bool have_elements_ = false, have_points_ = false, have_markers_ = false, have_periodic_ = false;
    std::int64_t npoin_ = 0;
    std::vector<int> element_lines_; // line of each element in data_.elements (deferred index check)
    std::int64_t marker_elements_ = 0;

    // ---------------------------------------------------------------------
    [[noreturn]] void fail_at(int line, const std::string& what) const {
        std::ostringstream os;
        os << source_ << ":";
        if (line > 0) os << " line " << line << ":";
        os << " " << what;
        throw MeshImportError(os.str());
    }
    [[noreturn]] void fail(const std::string& what) const { fail_at(line_no_, what); }

    // Next non-empty line without comment; false at the end of the file.
    bool next_line(std::string_view& out) {
        while (pos_ < buf_.size()) {
            std::size_t end = buf_.find('\n', pos_);
            if (end == std::string_view::npos) end = buf_.size();
            std::string_view line = buf_.substr(pos_, end - pos_);
            pos_ = end + 1;
            ++line_no_;
            const std::size_t comment = line.find('%');
            if (comment != std::string_view::npos) line = line.substr(0, comment);
            line = trim(line);
            if (!line.empty()) {
                out = line;
                return true;
            }
        }
        return false;
    }

    static bool split_keyword(std::string_view line, std::string& key, std::string_view& value) {
        const std::size_t eq = line.find('=');
        if (eq == std::string_view::npos) return false;
        key = upper(trim(line.substr(0, eq)));
        value = trim(line.substr(eq + 1));
        return !key.empty();
    }

    void require_ndime(const char* keyword) const {
        if (ndime_ == 0) fail(std::string("NDIME= must come before ") + keyword);
    }

    void require_once(bool& flag, const char* keyword) {
        if (flag) fail(std::string("duplicate ") + keyword + " section");
        require_ndime(keyword);
        flag = true;
    }

    // Non-negative count, bounded by the size of the rest of the file.
    std::int64_t count(std::string_view value, const std::string& what, std::size_t min_bytes_each = 0) const {
        std::int64_t n = 0;
        if (!parse_int(value, n)) fail("expected a count after " + what + ", found '" + excerpt(value) + "'");
        if (n < 0) fail("negative count after " + what + ": " + std::to_string(n));
        const std::size_t remaining = buf_.size() - std::min(pos_, buf_.size());
        if (min_bytes_each > 0 && static_cast<std::uint64_t>(n) > remaining / min_bytes_each + 1)
            fail(what + " " + std::to_string(n) + " exceeds the size of the file: file truncated or corrupt");
        return n;
    }

    // The next line of a section of `total` entries; `done` entries read.
    std::string_view section_line(const std::string& section, std::int64_t total, std::int64_t done) {
        std::string_view line;
        if (!next_line(line)) {
            fail_at(line_no_, "unexpected end of file: " + section + " " + std::to_string(total) + " declares " +
                                  std::to_string(total) + " lines, found " + std::to_string(done));
        }
        std::string key;
        std::string_view value;
        if (split_keyword(line, key, value)) {
            fail(section + " " + std::to_string(total) + " declares " + std::to_string(total) + " lines, found " +
                 std::to_string(done) + " before '" + key + "='");
        }
        return line;
    }

    void read_ndime(std::string_view value) {
        if (ndime_ != 0) fail("duplicate NDIME= line");
        std::int64_t d = 0;
        if (!parse_int(value, d)) fail("invalid NDIME= '" + excerpt(value) + "'");
        if (d == 3) fail("3D mesh (NDIME= 3): HydraMa2D needs a 2D mesh (NDIME= 2)");
        if (d != 2) fail("invalid dimension NDIME= " + std::to_string(d) + " (expected 2)");
        ndime_ = 2;
    }

    // Element line "type n_1 ... n_k [index]"; returns its type info.
    const VtkTypeInfo& element_line(std::string_view line, const std::string& section, RawElement& e) {
        const std::vector<std::string_view> tokens = split(line);
        std::int64_t type = 0;
        if (!parse_int(tokens.front(), type)) fail("expected a VTK element type in '" + excerpt(line) + "'");
        const VtkTypeInfo* info = find_vtk_type(type);
        if (!info) fail("unknown VTK element type " + std::to_string(type) + " in " + section);
        const std::size_t nn = static_cast<std::size_t>(info->nodes);
        if (tokens.size() != nn + 1 && tokens.size() != nn + 2) {
            fail("a " + std::string(info->name) + " (type " + std::to_string(type) + ") needs " + std::to_string(nn) +
                 " node indices (and an optional element index), found " + std::to_string(tokens.size() - 1) +
                 " values in '" + excerpt(line) + "'");
        }
        for (std::size_t k = 0; k < nn; ++k) {
            std::int64_t n = 0;
            if (!parse_int(tokens[k + 1], n)) fail("invalid node index '" + excerpt(tokens[k + 1]) + "'");
            if (n < 0 || n > std::numeric_limits<int>::max()) fail("node index " + std::to_string(n) + " out of range");
            if (k < 4) e.nodes[k] = n;
        }
        if (tokens.size() == nn + 2) {
            std::int64_t index = 0;
            if (!parse_int(tokens.back(), index)) fail("invalid element index '" + excerpt(tokens.back()) + "'");
        }
        return *info;
    }

    void read_elements(std::string_view value) {
        const std::int64_t n = count(value, "NELEM=", 6);
        data_.elements.reserve(data_.elements.size() + static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) {
            const std::string_view line = section_line("NELEM=", n, i);
            RawElement e;
            const VtkTypeInfo& info = element_line(line, "NELEM=", e);
            if (info.dim == 3) fail("a " + std::string(info.name) + " is a 3D element; HydraMa2D needs a 2D mesh");
            if (info.dim < 2) {
                fail("a " + std::string(info.name) + " (type " + std::to_string(info.type) +
                     ") is not a 2D element; NELEM= lists triangles (5) and quadrilaterals (9), boundary lines "
                     "belong to the markers");
            }
            e.type = info.type == 5 ? ElementType::Tri3 : ElementType::Quad4;
            e.tag = i;
            e.physical = 0;
            data_.elements.push_back(e);
            element_lines_.push_back(line_no_);
        }
    }

    void read_points(std::string_view value) {
        const std::vector<std::string_view> tokens = split(value);
        if (tokens.empty() || tokens.size() > 2) fail("expected 'NPOIN= count [domain count]', found '" + excerpt(value) + "'");
        const std::int64_t n = count(tokens[0], "NPOIN=", 4);
        if (tokens.size() == 2) {
            const std::int64_t domain = count(tokens[1], "NPOIN=");
            if (domain > n) fail("NPOIN= " + std::to_string(n) + " " + std::to_string(domain) +
                                 ": more domain points than points");
        }
        npoin_ = n;
        data_.node_tags.reserve(static_cast<std::size_t>(n));
        data_.x.reserve(static_cast<std::size_t>(n));
        data_.y.reserve(static_cast<std::size_t>(n));
        data_.z.reserve(static_cast<std::size_t>(n));
        std::int64_t mismatched = 0, first_mismatch = -1;
        for (std::int64_t i = 0; i < n; ++i) {
            const std::string_view line = section_line("NPOIN=", n, i);
            const std::vector<std::string_view> t = split(line);
            if (t.size() < 2 || t.size() > 3) {
                fail("a point of a 2D mesh is 'x y [index]', found " + std::to_string(t.size()) + " values in '" +
                     excerpt(line) + "'");
            }
            double x = 0.0, y = 0.0;
            if (!parse_double(t[0], x) || !parse_double(t[1], y))
                fail("expected finite coordinates 'x y [index]', found '" + excerpt(line) + "'");
            if (t.size() == 3) {
                std::int64_t index = 0;
                if (!parse_int(t[2], index))
                    fail("the third value of a 2D point must be its integer index, found '" + excerpt(t[2]) +
                         "' (3D coordinates in a NDIME= 2 file?)");
                if (index != i && mismatched++ == 0) first_mismatch = i;
            }
            data_.node_tags.push_back(i);
            data_.x.push_back(x);
            data_.y.push_back(y);
            data_.z.push_back(0.0);
        }
        if (mismatched > 0) {
            data_.warnings.push_back(std::to_string(mismatched) +
                                     " point index(es) differ from the position in the NPOIN= section (first: point " +
                                     std::to_string(first_mismatch) +
                                     "); as in SU2, points are numbered by position and the indices are ignored");
        }
    }

    void read_markers(std::string_view value) {
        const std::int64_t nmark = count(value, "NMARK=", 20);
        std::set<std::string> names;
        for (std::int64_t m = 0; m < nmark; ++m) {
            std::string_view line;
            std::string key;
            std::string_view name;
            if (!next_line(line)) fail("unexpected end of file: NMARK= " + std::to_string(nmark) + " declares " +
                                       std::to_string(nmark) + " markers, found " + std::to_string(m));
            if (!split_keyword(line, key, name) || key != "MARKER_TAG")
                fail("expected 'MARKER_TAG= name' of marker " + std::to_string(m + 1) + " of " + std::to_string(nmark) +
                     ", found '" + excerpt(line) + "'");
            if (name.empty()) fail("empty MARKER_TAG= name (marker " + std::to_string(m + 1) + ")");
            if (!names.insert(std::string(name)).second) fail("duplicate marker '" + std::string(name) + "'");
            const int physical = static_cast<int>(m + 1);
            data_.physical_names[{1, physical}] = std::string(name);

            std::string_view elems;
            if (!next_line(line) || !split_keyword(line, key, elems) || key != "MARKER_ELEMS")
                fail("expected 'MARKER_ELEMS= count' after 'MARKER_TAG= " + std::string(name) + "'");
            const std::int64_t k = count(elems, "MARKER_ELEMS=", 6);
            const std::string section = "MARKER_ELEMS= (marker '" + std::string(name) + "')";
            for (std::int64_t i = 0; i < k; ++i) {
                const std::string_view el = section_line(section, k, i);
                RawElement e;
                const VtkTypeInfo& info = element_line(el, section, e);
                if (info.type != 3) {
                    fail("marker '" + std::string(name) + "' contains a " + std::string(info.name) + " (type " +
                         std::to_string(info.type) + "); the markers of a 2D mesh contain lines (type 3)");
                }
                e.type = ElementType::Line2;
                e.physical = physical;
                e.tag = marker_elements_++;
                data_.elements.push_back(e);
                element_lines_.push_back(line_no_);
            }
        }
    }

    // Legacy (SU2 <= 6) periodic transformations: PERIODIC_INDEX= i followed
    // by three lines (centre, rotation angles, translation). Index 0 is the
    // identity that old SU2 versions always wrote.
    void read_periodic(std::string_view value) {
        const std::int64_t n = count(value, "NPERIODIC=", 20);
        bool nontrivial = false;
        for (std::int64_t p = 0; p < n; ++p) {
            std::string_view line;
            std::string key;
            std::string_view index;
            if (!next_line(line) || !split_keyword(line, key, index) || key != "PERIODIC_INDEX")
                fail("expected 'PERIODIC_INDEX= i' (" + std::to_string(p + 1) + " of NPERIODIC= " + std::to_string(n) + ")");
            std::int64_t i = 0;
            if (!parse_int(index, i)) fail("invalid PERIODIC_INDEX= '" + excerpt(index) + "'");
            for (int row = 0; row < 3; ++row) {
                const std::string_view data = section_line("PERIODIC_INDEX= " + std::to_string(i), 3, row);
                const std::vector<std::string_view> t = split(data);
                if (t.size() < 2 || t.size() > 3) fail("expected 2 or 3 values of a periodic transformation");
                for (std::string_view v : t) {
                    double d = 0.0;
                    if (!parse_double(v, d)) fail("invalid periodic transformation value '" + excerpt(v) + "'");
                    if (d != 0.0) nontrivial = true;
                }
            }
            if (i != 0) nontrivial = true;
        }
        if (nontrivial) {
            data_.warnings.push_back("periodic transformations (NPERIODIC=) ignored: periodic boundaries are not supported");
        }
    }

    void check_node_indices() const {
        for (std::size_t e = 0; e < data_.elements.size(); ++e) {
            const RawElement& el = data_.elements[e];
            for (int k = 0; k < node_count(el.type); ++k) {
                const std::int64_t n = el.nodes[static_cast<std::size_t>(k)];
                if (n >= npoin_) {
                    fail_at(element_lines_[e], std::string(el.type == ElementType::Line2 ? "boundary line" : "element") +
                                                   " references point " + std::to_string(n) + ", but NPOIN= " +
                                                   std::to_string(npoin_) + " (indices are 0-based)");
                }
            }
        }
    }
};

} // namespace

std::string_view su2_element_name(int vtk_type) {
    const VtkTypeInfo* info = find_vtk_type(vtk_type);
    return info ? info->name : std::string_view();
}

MeshData parse_su2(std::string_view contents, const std::string& source) { return Parser(contents, source).parse(); }

MeshData read_su2(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw MeshImportError(path + ": cannot open mesh file");
    std::ostringstream ss;
    ss << in.rdbuf();
    if (in.bad()) throw MeshImportError(path + ": read error");
    const std::string contents = ss.str();
    return parse_su2(contents, path);
}

} // namespace hydrama::mesh
