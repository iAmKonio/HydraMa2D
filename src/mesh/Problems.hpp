#pragma once

// Private helper of the mesh import code (not installed): collects the
// problems of a stage and throws them together as one MeshImportError, so that
// a user sees every problem of a category at once instead of one per run.
// Message format:
//   <source>: invalid mesh (N problems)
//     - first problem
//     ...
//     ... and M more

#include "mesh/MeshData.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace hydrama::mesh::detail {

class Problems {
public:
    Problems(std::string source, int limit) : source_(std::move(source)), limit_(std::max(limit, 1)) {}

    void add(const std::string& message) {
        if (static_cast<int>(messages_.size()) < limit_) messages_.push_back(message);
        ++count_;
    }

    int count() const { return count_; }

    void checkpoint() const {
        if (count_ == 0) return;
        std::ostringstream os;
        os << source_ << ": invalid mesh (" << count_ << (count_ == 1 ? " problem" : " problems") << ")";
        for (const std::string& m : messages_) os << "\n  - " << m;
        if (count_ > static_cast<int>(messages_.size()))
            os << "\n  ... and " << (count_ - static_cast<int>(messages_.size())) << " more";
        throw MeshImportError(os.str());
    }

private:
    std::string source_;
    int limit_;
    int count_ = 0;
    std::vector<std::string> messages_;
};

} // namespace hydrama::mesh::detail
