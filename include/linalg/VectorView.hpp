#pragma once

// Non-owning views of contiguous double arrays (a minimal C++17 stand-in for
// std::span). Every linear algebra interface takes views rather than
// std::vector so that operators, preconditioners and Krylov solvers can act
// on any contiguous storage (whole fields today, sub-blocks such as (u, v)
// components or permuted work arrays later) without copies.
//
//   VectorView       mutable view   (output arguments)
//   ConstVectorView  read-only view (input arguments)
//
// A VectorView converts implicitly to a ConstVectorView, and both convert
// implicitly from std::vector<double>.

#include <cstddef>
#include <type_traits>
#include <vector>

namespace hydrama::linalg {

template <class T>
class Span {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;

    constexpr Span() noexcept = default;
    constexpr Span(T* data, std::size_t size) noexcept : data_(data), size_(size) {}

    // From a mutable vector (always allowed).
    Span(std::vector<value_type>& v) noexcept : data_(v.data()), size_(v.size()) {}

    // From a const vector (read-only views only).
    template <class U = T, class = std::enable_if_t<std::is_const_v<U>>>
    Span(const std::vector<value_type>& v) noexcept : data_(v.data()), size_(v.size()) {}

    // Mutable view -> read-only view.
    template <class U, class = std::enable_if_t<std::is_const_v<T> && std::is_same_v<const U, T>>>
    constexpr Span(const Span<U>& other) noexcept : data_(other.data()), size_(other.size()) {}

    constexpr T* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr T& operator[](std::size_t i) const noexcept { return data_[i]; }

    constexpr T* begin() const noexcept { return data_; }
    constexpr T* end() const noexcept { return data_ + size_; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

using VectorView = Span<double>;
using ConstVectorView = Span<const double>;

} // namespace hydrama::linalg
