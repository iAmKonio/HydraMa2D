#include "mesh/GmshReader.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <type_traits>

namespace hydrama::mesh {

namespace {

struct GmshTypeInfo {
    int type;
    int nodes;
    int dim;
    bool higher_order;
    std::string_view name;
};

// Gmsh element types (gmsh reference manual, "MSH file format").
constexpr GmshTypeInfo kGmshTypes[] = {
    {1, 2, 1, false, "2-node line"},
    {2, 3, 2, false, "3-node triangle"},
    {3, 4, 2, false, "4-node quadrangle"},
    {4, 4, 3, false, "4-node tetrahedron"},
    {5, 8, 3, false, "8-node hexahedron"},
    {6, 6, 3, false, "6-node prism"},
    {7, 5, 3, false, "5-node pyramid"},
    {8, 3, 1, true, "3-node second-order line"},
    {9, 6, 2, true, "6-node second-order triangle"},
    {10, 9, 2, true, "9-node second-order quadrangle"},
    {11, 10, 3, true, "10-node second-order tetrahedron"},
    {12, 27, 3, true, "27-node second-order hexahedron"},
    {13, 18, 3, true, "18-node second-order prism"},
    {14, 14, 3, true, "14-node second-order pyramid"},
    {15, 1, 0, false, "1-node point"},
    {16, 8, 2, true, "8-node second-order quadrangle"},
    {17, 20, 3, true, "20-node second-order hexahedron"},
    {18, 15, 3, true, "15-node second-order prism"},
    {19, 13, 3, true, "13-node second-order pyramid"},
    {20, 9, 2, true, "9-node third-order incomplete triangle"},
    {21, 10, 2, true, "10-node third-order triangle"},
    {22, 12, 2, true, "12-node fourth-order incomplete triangle"},
    {23, 15, 2, true, "15-node fourth-order triangle"},
    {24, 15, 2, true, "15-node fifth-order incomplete triangle"},
    {25, 21, 2, true, "21-node fifth-order triangle"},
    {26, 4, 1, true, "4-node third-order line"},
    {27, 5, 1, true, "5-node fourth-order line"},
    {28, 6, 1, true, "6-node fifth-order line"},
    {29, 20, 3, true, "20-node third-order tetrahedron"},
    {30, 35, 3, true, "35-node fourth-order tetrahedron"},
    {31, 56, 3, true, "56-node fifth-order tetrahedron"},
    {92, 64, 3, true, "64-node third-order hexahedron"},
    {93, 125, 3, true, "125-node fourth-order hexahedron"},
};

const GmshTypeInfo* find_type(int type) {
    for (const GmshTypeInfo& t : kGmshTypes)
        if (t.type == type) return &t;
    return nullptr;
}

template <typename T>
T byte_swapped(T value) {
    unsigned char bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));
    std::reverse(bytes, bytes + sizeof(T));
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v'; }

class Reader {
public:
    Reader(std::string_view buf, std::string source) : buf_(buf), source_(std::move(source)) {
        data_.source = source_;
    }

    MeshData parse() {
        bool have_nodes = false, have_elements = false, have_entities = false;
        while (true) {
            skip_space();
            if (eof()) break;
            const std::size_t header_pos = pos_;
            const std::string_view line = read_line();
            if (line.empty()) continue;
            if (line.front() != '$') {
                pos_ = header_pos;
                fail("expected a section header ($Name), found '" + excerpt(line) + "'");
            }
            const std::string name(line.substr(1));
            if (!have_format_) {
                if (name == "NOD" || name == "NOE" || name == "ELM") {
                    fail("this is an MSH 1.0 file, which is not supported; save the mesh as MSH 4.1 or 2.2 "
                         "(gmsh option Mesh.MshFileVersion)");
                }
                if (name != "MeshFormat") fail("not a Gmsh MSH file: the first section must be $MeshFormat, found $" + name);
                read_mesh_format();
                have_format_ = true;
            } else if (name == "MeshFormat") {
                fail("duplicate $MeshFormat section");
            } else if (name == "PhysicalNames") {
                read_physical_names();
            } else if (name == "Entities" && version_ == 4) {
                if (have_entities) fail("duplicate $Entities section");
                read_entities();
                have_entities = true;
            } else if (name == "PartitionedEntities") {
                fail("partitioned meshes ($PartitionedEntities) are not supported; save the mesh unpartitioned");
            } else if (name == "Nodes") {
                if (have_nodes) fail("duplicate $Nodes section");
                if (version_ == 2) read_nodes_v2(); else read_nodes_v4();
                have_nodes = true;
            } else if (name == "Elements") {
                if (have_elements) fail("duplicate $Elements section");
                if (!have_nodes) fail("$Elements before $Nodes");
                if (version_ == 2) read_elements_v2(); else read_elements_v4();
                have_elements = true;
            } else {
                skip_section(name);
            }
        }
        if (!have_format_) fail("empty file: not a Gmsh MSH file");
        if (!have_nodes) fail("no $Nodes section");
        if (!have_elements) fail("no $Elements section");
        expand_entity_physicals();
        return std::move(data_);
    }

private:
    std::string_view buf_;
    std::size_t pos_ = 0;
    std::string source_;
    MeshData data_;

    bool have_format_ = false;
    int version_ = 0;          // 2 or 4
    bool binary_ = false;
    bool swap_ = false;        // binary data in the other byte order
    int size_t_bytes_ = 8;     // MSH 4.1 binary: width of size_t fields

    // MSH 4.1: (dim, entity tag) -> physical tags, from $Entities.
    std::map<std::pair<int, int>, std::vector<int>> entity_physicals_;
    // MSH 4.1: elements whose physical tags come from their entity.
    struct PendingEntity {
        std::size_t first, last; // element range in data_.elements
        int dim, entity;
    };
    std::vector<PendingEntity> pending_;

    // ---------------------------------------------------------------------
    // Errors
    // ---------------------------------------------------------------------
    [[noreturn]] void fail(const std::string& what) const {
        std::ostringstream os;
        os << source_ << ":";
        if (binary_) {
            os << " byte " << pos_;
        } else {
            os << " line " << (1 + std::count(buf_.begin(), buf_.begin() + static_cast<std::ptrdiff_t>(std::min(pos_, buf_.size())), '\n'));
        }
        os << ": " << what;
        throw MeshImportError(os.str());
    }

    static std::string excerpt(std::string_view s) {
        std::string out(s.substr(0, 60));
        for (char& c : out)
            if (static_cast<unsigned char>(c) < 32 || static_cast<unsigned char>(c) > 126) c = '?';
        if (s.size() > 60) out += "...";
        return out;
    }

    // ---------------------------------------------------------------------
    // Low-level reading
    // ---------------------------------------------------------------------
    bool eof() const { return pos_ >= buf_.size(); }

    void skip_space() {
        while (pos_ < buf_.size() && is_space(buf_[pos_])) ++pos_;
    }

    // Up to (excluding) the next '\n' or the end; trailing blanks and '\r'
    // removed. Consumes the '\n'.
    std::string_view read_line() {
        const std::size_t start = pos_;
        std::size_t end = buf_.find('\n', pos_);
        if (end == std::string_view::npos) end = buf_.size();
        pos_ = std::min(end + 1, buf_.size());
        std::string_view line = buf_.substr(start, end - start);
        while (!line.empty() && is_space(line.back())) line.remove_suffix(1);
        while (!line.empty() && is_space(line.front())) line.remove_prefix(1);
        return line;
    }

    std::string_view token(const char* what) {
        skip_space();
        if (eof()) fail(std::string("unexpected end of file while reading ") + what);
        const std::size_t start = pos_;
        while (pos_ < buf_.size() && !is_space(buf_[pos_])) ++pos_;
        return buf_.substr(start, pos_ - start);
    }

    static bool parse_int(std::string_view s, std::int64_t& out) {
        if (!s.empty() && s.front() == '+') s.remove_prefix(1);
        const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
        return r.ec == std::errc() && r.ptr == s.data() + s.size();
    }

    static bool parse_double(std::string_view s, double& out) {
        if (!s.empty() && s.front() == '+') s.remove_prefix(1);
        const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
        return r.ec == std::errc() && r.ptr == s.data() + s.size() && std::isfinite(out);
    }

    std::int64_t ascii_int(const char* what) {
        const std::string_view t = token(what);
        std::int64_t v = 0;
        if (!parse_int(t, v)) fail(std::string("expected an integer (") + what + "), found '" + excerpt(t) + "'");
        return v;
    }

    int ascii_int32(const char* what) {
        const std::int64_t v = ascii_int(what);
        if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
            fail(std::string("value out of range (") + what + "): " + std::to_string(v));
        return static_cast<int>(v);
    }

    std::int64_t ascii_count(const char* what) {
        const std::int64_t v = ascii_int(what);
        if (v < 0) fail(std::string("negative count (") + what + "): " + std::to_string(v));
        return v;
    }

    double ascii_double(const char* what) {
        const std::string_view t = token(what);
        double v = 0.0;
        if (!parse_double(t, v)) fail(std::string("expected a finite real number (") + what + "), found '" + excerpt(t) + "'");
        return v;
    }

    template <typename T>
    T binary_value(const char* what) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (pos_ > buf_.size() || buf_.size() - pos_ < sizeof(T))
            fail(std::string("unexpected end of file in binary data (") + what + "): file truncated");
        T v;
        std::memcpy(&v, buf_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return swap_ ? byte_swapped(v) : v;
    }

    int binary_int(const char* what) { return binary_value<std::int32_t>(what); }

    double binary_double(const char* what) {
        const double v = binary_value<double>(what);
        if (!std::isfinite(v)) fail(std::string("non-finite real number (") + what + ")");
        return v;
    }

    // MSH 4.1 size_t field (4 or 8 bytes).
    std::uint64_t binary_size(const char* what) {
        if (size_t_bytes_ == 4) return binary_value<std::uint32_t>(what);
        return binary_value<std::uint64_t>(what);
    }

    // Counts and tags that are size_t in MSH 4.1 (binary) / integers (ASCII).
    std::int64_t v4_size(const char* what) {
        if (!binary_) return ascii_count(what);
        const std::uint64_t v = binary_size(what);
        if (v > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            fail(std::string("value out of range (") + what + ")");
        return static_cast<std::int64_t>(v);
    }

    int v4_int(const char* what) { return binary_ ? binary_int(what) : ascii_int32(what); }
    double v4_double(const char* what) { return binary_ ? binary_double(what) : ascii_double(what); }

    // Rejects counts that cannot fit in the rest of the file (corrupt or
    // truncated headers must not trigger huge allocations).
    void check_count(std::int64_t count, std::size_t min_bytes_each, const char* what) {
        const std::size_t remaining = buf_.size() - std::min(pos_, buf_.size());
        if (count < 0 || static_cast<std::uint64_t>(count) > remaining / std::max<std::size_t>(min_bytes_each, 1) + 1)
            fail(std::string("count ") + std::to_string(count) + " of " + what +
                 " exceeds the size of the file: file truncated or corrupt");
    }

    // MSH 2 binary: the counts of $Nodes / $Elements are a text line and the
    // binary data starts right after its '\n'. (MSH 4.1 binary sections are
    // binary from the first byte after the section header: no text line.)
    void finish_text_line() {
        while (pos_ < buf_.size() && (buf_[pos_] == ' ' || buf_[pos_] == '\t' || buf_[pos_] == '\r')) ++pos_;
        if (pos_ < buf_.size() && buf_[pos_] == '\n') ++pos_;
    }

    void expect_end(const std::string& name) {
        skip_space();
        const std::size_t at = pos_;
        const std::string_view line = read_line();
        if (line != "$End" + name) {
            pos_ = at;
            fail("expected $End" + name + ", found '" + excerpt(line) + "'" +
                 (binary_ ? " (binary data does not match the declared counts)" : ""));
        }
    }

    void skip_section(const std::string& name) {
        const std::string end = "$End" + name;
        std::size_t at = pos_;
        while (true) {
            at = buf_.find(end, at);
            if (at == std::string_view::npos) fail("section $" + name + " has no " + end);
            const bool line_start = at == 0 || buf_[at - 1] == '\n';
            const std::size_t after = at + end.size();
            const bool line_end = after == buf_.size() || is_space(buf_[after]);
            if (line_start && line_end) break;
            at = after;
        }
        pos_ = at;
        read_line();
    }

    // ---------------------------------------------------------------------
    // Sections
    // ---------------------------------------------------------------------
    void read_mesh_format() {
        skip_space();
        const std::string_view line = read_line();
        std::istringstream is{std::string(line)};
        std::string version;
        int file_type = -1, data_size = -1;
        if (!(is >> version >> file_type >> data_size)) fail("invalid $MeshFormat line '" + excerpt(line) + "'");

        if (version == "2" || version == "2.0" || version == "2.1" || version == "2.2") {
            version_ = 2;
        } else if (version == "4.1") {
            version_ = 4;
        } else if (version == "4" || version == "4.0") {
            fail("MSH 4.0 is not supported; save the mesh as MSH 4.1 or 2.2 (gmsh option Mesh.MshFileVersion)");
        } else {
            fail("unsupported MSH version " + version + "; supported are 4.1 and 2.2");
        }
        if (file_type != 0 && file_type != 1) fail("invalid file type " + std::to_string(file_type) + " in $MeshFormat (0 = ASCII, 1 = binary)");
        binary_ = file_type == 1;
        data_.format = "Gmsh MSH " + version + (binary_ ? " binary" : " ASCII");

        if (binary_) {
            if (version_ == 2 && data_size != static_cast<int>(sizeof(double)))
                fail("unsupported data size " + std::to_string(data_size) + " in binary MSH 2 (expected 8)");
            if (version_ == 4 && data_size != 4 && data_size != 8)
                fail("unsupported data size " + std::to_string(data_size) + " in binary MSH 4.1 (expected 4 or 8)");
            size_t_bytes_ = data_size;
            // Endianness marker: the integer 1 written in the file's byte order.
            const std::int32_t one = binary_value<std::int32_t>("endianness marker");
            if (one == 1) {
                swap_ = false;
            } else if (byte_swapped(one) == 1) {
                swap_ = true;
            } else {
                fail("invalid endianness marker in binary $MeshFormat");
            }
            finish_text_line();
        }
        expect_end("MeshFormat");
    }

    void read_physical_names() {
        skip_space();
        const std::int64_t count = ascii_count("number of physical names");
        check_count(count, 6, "physical names");
        read_line();
        for (std::int64_t i = 0; i < count; ++i) {
            skip_space();
            const std::string_view line = read_line();
            std::size_t p = 0;
            auto next_word = [&]() {
                while (p < line.size() && is_space(line[p])) ++p;
                const std::size_t s = p;
                while (p < line.size() && !is_space(line[p])) ++p;
                return line.substr(s, p - s);
            };
            std::int64_t dim = 0, tag = 0;
            if (!parse_int(next_word(), dim) || !parse_int(next_word(), tag) || dim < 0 || dim > 3)
                fail("invalid physical name entry '" + excerpt(line) + "' (expected: dim tag \"name\")");
            std::string_view rest = line.substr(p);
            while (!rest.empty() && is_space(rest.front())) rest.remove_prefix(1);
            std::string name;
            if (!rest.empty() && rest.front() == '"') {
                const std::size_t close = rest.rfind('"');
                if (close == 0) fail("unterminated quoted physical name in '" + excerpt(line) + "'");
                name = std::string(rest.substr(1, close - 1));
            } else {
                name = std::string(rest);
            }
            const auto key = std::make_pair(static_cast<int>(dim), static_cast<int>(tag));
            if (!data_.physical_names.emplace(key, name).second)
                fail("duplicate physical name for dimension " + std::to_string(dim) + ", tag " + std::to_string(tag));
        }
        expect_end("PhysicalNames");
    }

    void read_entities() {
        std::int64_t counts[4];
        for (std::int64_t& c : counts) c = v4_size("number of entities");
        for (int dim = 0; dim < 4; ++dim) {
            check_count(counts[dim], binary_ ? 12 : 8, "entities");
            for (std::int64_t i = 0; i < counts[dim]; ++i) {
                const int tag = v4_int("entity tag");
                const int ncoords = dim == 0 ? 3 : 6; // point: X Y Z; others: bounding box
                for (int k = 0; k < ncoords; ++k) v4_double("entity coordinate");
                const std::int64_t nphys = v4_size("number of physical tags");
                check_count(nphys, binary_ ? 4 : 2, "physical tags");
                std::vector<int> phys;
                for (std::int64_t k = 0; k < nphys; ++k) {
                    const int p = v4_int("physical tag");
                    if (p != 0 && std::find(phys.begin(), phys.end(), p) == phys.end()) phys.push_back(p);
                }
                if (dim > 0) {
                    const std::int64_t nbound = v4_size("number of bounding entities");
                    check_count(nbound, binary_ ? 4 : 2, "bounding entities");
                    for (std::int64_t k = 0; k < nbound; ++k) v4_int("bounding entity tag");
                }
                if (!entity_physicals_.emplace(std::make_pair(dim, tag), std::move(phys)).second)
                    fail("duplicate entity: dimension " + std::to_string(dim) + ", tag " + std::to_string(tag));
            }
        }
        expect_end("Entities");
    }

    void add_node(std::int64_t tag, double x, double y, double z) {
        data_.node_tags.push_back(tag);
        data_.x.push_back(x);
        data_.y.push_back(y);
        data_.z.push_back(z);
    }

    void reserve_nodes(std::int64_t n) {
        data_.node_tags.reserve(static_cast<std::size_t>(n));
        data_.x.reserve(static_cast<std::size_t>(n));
        data_.y.reserve(static_cast<std::size_t>(n));
        data_.z.reserve(static_cast<std::size_t>(n));
    }

    void read_nodes_v2() {
        skip_space();
        const std::int64_t count = ascii_count("number of nodes");
        if (binary_) finish_text_line();
        check_count(count, binary_ ? 28 : 8, "nodes");
        reserve_nodes(count);
        for (std::int64_t i = 0; i < count; ++i) {
            if (binary_) {
                const int tag = binary_int("node tag");
                const double x = binary_double("node x");
                const double y = binary_double("node y");
                const double z = binary_double("node z");
                add_node(tag, x, y, z);
            } else {
                const std::int64_t tag = ascii_int("node tag");
                const double x = ascii_double("node x");
                const double y = ascii_double("node y");
                const double z = ascii_double("node z");
                add_node(tag, x, y, z);
            }
        }
        expect_end("Nodes");
    }

    void read_nodes_v4() {
        const std::int64_t blocks = v4_size("number of node blocks");
        const std::int64_t total = v4_size("number of nodes");
        v4_size("minimum node tag");
        v4_size("maximum node tag");
        const std::size_t st = static_cast<std::size_t>(size_t_bytes_);
        check_count(blocks, binary_ ? 12 + st : 8, "node blocks");
        check_count(total, binary_ ? st + 24 : 8, "nodes");
        reserve_nodes(total);
        std::vector<std::int64_t> tags;
        for (std::int64_t b = 0; b < blocks; ++b) {
            const int dim = v4_int("entity dimension");
            v4_int("entity tag");
            const int parametric = v4_int("parametric flag");
            const std::int64_t n = v4_size("number of nodes in block");
            if (dim < 0 || dim > 3) fail("invalid entity dimension " + std::to_string(dim) + " in node block");
            if (parametric != 0 && parametric != 1) fail("invalid parametric flag " + std::to_string(parametric));
            check_count(n, binary_ ? st + 24 : 8, "nodes in block");
            if (static_cast<std::int64_t>(data_.node_tags.size()) + n > total)
                fail("node blocks contain more nodes than the declared total " + std::to_string(total));
            const int extra = parametric ? std::min(dim, 3) : 0; // u (dim>=1), v (dim>=2), w (dim==3)
            tags.resize(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i) tags[static_cast<std::size_t>(i)] = v4_size("node tag");
            for (std::int64_t i = 0; i < n; ++i) {
                const double x = v4_double("node x");
                const double y = v4_double("node y");
                const double z = v4_double("node z");
                for (int k = 0; k < extra; ++k) v4_double("parametric coordinate");
                add_node(tags[static_cast<std::size_t>(i)], x, y, z);
            }
        }
        if (static_cast<std::int64_t>(data_.node_tags.size()) != total)
            fail("node blocks contain " + std::to_string(data_.node_tags.size()) + " nodes, the header declares " +
                 std::to_string(total));
        expect_end("Nodes");
    }

    // Element type of a Gmsh type number; nullopt for ignored types (points).
    // Fails for unsupported types.
    std::optional<ElementType> element_type(int gmsh_type) {
        switch (gmsh_type) {
        case 1: return ElementType::Line2;
        case 2: return ElementType::Tri3;
        case 3: return ElementType::Quad4;
        case 15: return std::nullopt;
        default: break;
        }
        const GmshTypeInfo* info = find_type(gmsh_type);
        if (!info) fail("unknown Gmsh element type " + std::to_string(gmsh_type));
        const std::string what = "element type " + std::to_string(gmsh_type) + " (" + std::string(info->name) + ")";
        if (info->dim == 3) fail(what + " is a 3D element; HydraMa2D needs a 2D mesh (e.g. gmsh -2)");
        fail(what + " is a higher-order element; HydraMa2D needs linear elements (gmsh option Mesh.ElementOrder = 1)");
    }

    void read_elements_v2() {
        skip_space();
        const std::int64_t count = ascii_count("number of elements");
        if (binary_) finish_text_line();
        check_count(count, binary_ ? 12 : 6, "elements");
        data_.elements.reserve(static_cast<std::size_t>(count));
        std::int64_t done = 0;
        std::vector<std::int64_t> tags;
        if (!binary_) {
            for (; done < count; ++done) {
                const std::int64_t tag = ascii_int("element tag");
                const int type = ascii_int32("element type");
                const std::int64_t ntags = ascii_count("number of element tags");
                check_count(ntags, 2, "element tags");
                tags.resize(static_cast<std::size_t>(ntags));
                for (std::int64_t& t : tags) t = ascii_int("element tag value");
                const std::optional<ElementType> et = element_type(type);
                const int nn = find_type(type)->nodes;
                RawElement e;
                e.tag = tag;
                e.physical = ntags > 0 ? static_cast<int>(tags[0]) : 0;
                e.entity = ntags > 1 ? static_cast<int>(tags[1]) : 0;
                for (int k = 0; k < nn; ++k) {
                    const std::int64_t node = ascii_int("element node");
                    if (k < 4) e.nodes[static_cast<std::size_t>(k)] = node;
                }
                if (et) {
                    e.type = *et;
                    data_.elements.push_back(e);
                }
            }
        } else {
            while (done < count) {
                const int type = binary_int("element type");
                const int follow = binary_int("number of elements in group");
                const int ntags = binary_int("number of element tags");
                if (follow < 0 || ntags < 0 || follow > count - done)
                    fail("invalid binary element group header (type " + std::to_string(type) + ", " +
                         std::to_string(follow) + " elements, " + std::to_string(ntags) + " tags)");
                const std::optional<ElementType> et = element_type(type);
                const int nn = find_type(type)->nodes;
                check_count(follow, 4u * static_cast<std::size_t>(1 + ntags + nn), "elements in group");
                for (int i = 0; i < follow; ++i) {
                    RawElement e;
                    e.tag = binary_int("element tag");
                    for (int k = 0; k < ntags; ++k) {
                        const int t = binary_int("element tag value");
                        if (k == 0) e.physical = t;
                        if (k == 1) e.entity = t;
                    }
                    for (int k = 0; k < nn; ++k) {
                        const int node = binary_int("element node");
                        if (k < 4) e.nodes[static_cast<std::size_t>(k)] = node;
                    }
                    if (et) {
                        e.type = *et;
                        data_.elements.push_back(e);
                    }
                }
                done += follow;
            }
        }
        expect_end("Elements");
    }

    void read_elements_v4() {
        const std::int64_t blocks = v4_size("number of element blocks");
        const std::int64_t total = v4_size("number of elements");
        v4_size("minimum element tag");
        v4_size("maximum element tag");
        const std::size_t st = static_cast<std::size_t>(size_t_bytes_);
        check_count(blocks, binary_ ? 12 + st : 8, "element blocks");
        check_count(total, binary_ ? 2 * st : 4, "elements"); // smallest: a point (tag + 1 node)
        data_.elements.reserve(static_cast<std::size_t>(total));
        std::int64_t seen = 0;
        for (std::int64_t b = 0; b < blocks; ++b) {
            const int dim = v4_int("entity dimension");
            const int entity = v4_int("entity tag");
            const int type = v4_int("element type");
            const std::int64_t n = v4_size("number of elements in block");
            const std::optional<ElementType> et = element_type(type);
            const GmshTypeInfo* info = find_type(type);
            if (info->dim != dim)
                fail("element block of entity dimension " + std::to_string(dim) + " contains " +
                     std::string(info->name) + " elements");
            const int nn = info->nodes;
            check_count(n, binary_ ? static_cast<std::size_t>(size_t_bytes_) * static_cast<std::size_t>(nn + 1)
                                   : 2u * static_cast<std::size_t>(nn + 1),
                        "elements in block");
            seen += n;
            if (seen > total) fail("element blocks contain more elements than the declared total " + std::to_string(total));
            const std::size_t first = data_.elements.size();
            for (std::int64_t i = 0; i < n; ++i) {
                RawElement e;
                e.tag = v4_size("element tag");
                e.entity = entity;
                for (int k = 0; k < nn; ++k) {
                    const std::int64_t node = v4_size("element node");
                    if (k < 4) e.nodes[static_cast<std::size_t>(k)] = node;
                }
                if (et) {
                    e.type = *et;
                    data_.elements.push_back(e);
                }
            }
            if (et && data_.elements.size() > first) pending_.push_back({first, data_.elements.size(), dim, entity});
        }
        if (seen != total)
            fail("element blocks contain " + std::to_string(seen) + " elements, the header declares " + std::to_string(total));
        expect_end("Elements");
    }

    // MSH 4.1: physical tags come from the entity of each element block.
    // An element of an entity in k > 1 physical groups is stored k times
    // (once per group), as in MSH 2.2.
    void expand_entity_physicals() {
        if (version_ != 4) return;
        std::vector<RawElement> out;
        out.reserve(data_.elements.size());
        for (const PendingEntity& block : pending_) {
            const auto it = entity_physicals_.find({block.dim, block.entity});
            static const std::vector<int> none;
            const std::vector<int>& phys = it == entity_physicals_.end() ? none : it->second;
            for (std::size_t i = block.first; i < block.last; ++i) {
                RawElement e = data_.elements[i];
                if (phys.empty()) {
                    out.push_back(e);
                } else {
                    for (int p : phys) {
                        e.physical = p;
                        out.push_back(e);
                    }
                }
            }
        }
        data_.elements = std::move(out);
    }
};

} // namespace

std::string_view gmsh_element_name(int type) {
    const GmshTypeInfo* info = find_type(type);
    return info ? info->name : std::string_view();
}

MeshData parse_gmsh(std::string_view contents, const std::string& source) {
    return Reader(contents, source).parse();
}

MeshData read_gmsh(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw MeshImportError(path + ": cannot open mesh file");
    std::ostringstream ss;
    ss << in.rdbuf();
    if (in.bad()) throw MeshImportError(path + ": read error");
    const std::string contents = ss.str();
    return parse_gmsh(contents, path);
}

} // namespace hydrama::mesh
