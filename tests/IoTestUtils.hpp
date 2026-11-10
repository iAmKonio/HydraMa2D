#pragma once

// File-system helpers for IO and solver tests: unique temporary directories,
// text file utilities, stdout capture, a case copier and a parser for the
// legacy VTK files written by IO::write_vtk.

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hydrama::test {

namespace fs = std::filesystem;

// Unique directory below the system temp directory, removed on destruction.
// Unique per process and call, so tests can run in parallel (ctest -j).
class TempDir {
public:
    explicit TempDir(const std::string& tag = "tmp") {
        static std::atomic<unsigned> counter{0};
        std::random_device rd;
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() / "hydrama2d-tests" /
                (tag + "-" + std::to_string(stamp) + "-" + std::to_string(rd()) + "-" +
                 std::to_string(counter++));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const fs::path& path() const { return path_; }
    fs::path operator/(const std::string& name) const { return path_ / name; }

private:
    fs::path path_;
};

inline void write_text(const fs::path& file, const std::string& content) {
    std::ofstream out(file, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + file.string());
    out << content;
}

inline std::string read_text(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + file.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}

inline bool is_blank(const std::string& s) { return s.find_first_not_of(" \t\r") == std::string::npos; }

// Replace the k-th NON-EMPTY line (0-based, the indexing used by the IO
// config readers) of a text file, keeping blank lines and line endings.
inline void set_nonempty_lines(const fs::path& file, const std::map<int, std::string>& values) {
    const std::string text = read_text(file);
    const bool crlf = text.find("\r\n") != std::string::npos;
    std::vector<std::string> lines = split_lines(text);
    int k = 0;
    for (auto& line : lines) {
        if (is_blank(line)) continue;
        auto it = values.find(k++);
        if (it != values.end()) line = it->second + (crlf ? "\r" : "");
    }
    std::string out;
    for (const auto& line : lines) out += line + "\n";
    write_text(file, out);
}

// Redirects an output stream (std::cout / std::cerr) into a string for the
// lifetime of the object and restores its formatting state afterwards (the
// solver switches std::cout to std::scientific, which would otherwise leak
// into later output).
class CaptureStream {
public:
    explicit CaptureStream(std::ostream& stream)
        : stream_(stream), old_(stream.rdbuf(buffer_.rdbuf())), flags_(stream.flags()),
          precision_(stream.precision()) {}
    ~CaptureStream() {
        stream_.rdbuf(old_);
        stream_.flags(flags_);
        stream_.precision(precision_);
    }
    CaptureStream(const CaptureStream&) = delete;
    CaptureStream& operator=(const CaptureStream&) = delete;
    std::string str() const { return buffer_.str(); }

private:
    std::ostream& stream_;
    std::ostringstream buffer_;
    std::streambuf* old_;
    std::ios::fmtflags flags_;
    std::streamsize precision_;
};

struct CaptureStdout : CaptureStream {
    CaptureStdout() : CaptureStream(std::cout) {}
};

struct CaptureStderr : CaptureStream {
    CaptureStderr() : CaptureStream(std::cerr) {}
};

// Copy constant/ and system/ of the bundled benchmark case into dest.
inline void copy_benchmark_case(const fs::path& dest) {
    const fs::path src = HYDRAMA2D_CASE_DIR;
    fs::create_directories(dest);
    fs::copy(src / "constant", dest / "constant", fs::copy_options::recursive);
    fs::copy(src / "system", dest / "system", fs::copy_options::recursive);
}

// Minimal parser for the legacy ASCII VTK unstructured grid written by
// IO::write_vtk (points, triangles, point vectors "velocity", scalars "P",
// followed by any number of additional point scalars).
struct VtkFile {
    std::vector<std::string> header; // first four lines
    int npoints = 0, ncells = 0, cells_size = 0, ntypes = 0, npointdata = 0;
    std::vector<std::array<double, 3>> points;
    std::vector<std::array<int, 4>> cells; // {3, a, b, c}
    std::vector<int> cell_types;
    std::vector<std::array<double, 3>> velocity;
    std::vector<double> pressure;
    std::string vectors_line, scalars_line, lookup_line;
    // Additional "SCALARS <name> float 1" sections after P, in file order.
    std::vector<std::pair<std::string, std::vector<double>>> extra_scalars;

    const std::vector<double>* scalar(const std::string& name) const {
        for (const auto& s : extra_scalars)
            if (s.first == name) return &s.second;
        return nullptr;
    }
};

inline VtkFile parse_vtk(const fs::path& file) {
    std::istringstream in(read_text(file));
    VtkFile v;
    std::string line, word;
    for (int i = 0; i < 4 && std::getline(in, line); ++i) v.header.push_back(line);

    auto expect = [&](const std::string& keyword) {
        in >> word;
        if (word != keyword) throw std::runtime_error("VTK: expected " + keyword + ", got " + word);
    };
    expect("POINTS");
    in >> v.npoints >> word; // "double"
    v.points.resize(v.npoints);
    for (auto& p : v.points) in >> p[0] >> p[1] >> p[2];

    expect("CELLS");
    in >> v.ncells >> v.cells_size;
    v.cells.resize(v.ncells);
    for (auto& c : v.cells) in >> c[0] >> c[1] >> c[2] >> c[3];

    expect("CELL_TYPES");
    in >> v.ntypes;
    v.cell_types.resize(v.ntypes);
    for (auto& t : v.cell_types) in >> t;

    expect("POINT_DATA");
    in >> v.npointdata;
    std::getline(in, line); // rest of POINT_DATA line
    std::getline(in, v.vectors_line);
    v.velocity.resize(v.npointdata);
    for (auto& u : v.velocity) in >> u[0] >> u[1] >> u[2];
    std::getline(in, line); // rest of last vector line
    std::getline(in, v.scalars_line);
    std::getline(in, v.lookup_line);
    v.pressure.resize(v.npointdata);
    for (auto& p : v.pressure) in >> p;
    if (!in) throw std::runtime_error("VTK: truncated file " + file.string());

    while (in >> word) {
        if (word != "SCALARS") throw std::runtime_error("VTK: expected SCALARS, got " + word);
        std::string name, type;
        int ncomp = 0;
        in >> name >> type >> ncomp;
        if (type != "float" || ncomp != 1) throw std::runtime_error("VTK: unsupported scalar section " + name);
        expect("LOOKUP_TABLE");
        in >> word; // "default"
        std::vector<double> values(v.npointdata);
        for (auto& x : values) in >> x;
        if (!in) throw std::runtime_error("VTK: truncated scalar section " + name);
        v.extra_scalars.emplace_back(name, std::move(values));
    }
    return v;
}

} // namespace hydrama::test
