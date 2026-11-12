#pragma once

#if defined(HYDRAMA2D_HAVE_OPENMP) && defined(_OPENMP)
#include <omp.h>
#define HYDRAMA2D_OPENMP 1
#endif

namespace hydrama {

inline int get_max_threads() noexcept {
#ifdef HYDRAMA2D_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int get_thread_num() noexcept {
#ifdef HYDRAMA2D_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline void set_num_threads(int n) noexcept {
#ifdef HYDRAMA2D_OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

inline int get_num_threads() noexcept {
#ifdef HYDRAMA2D_OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

inline bool in_parallel() noexcept {
#ifdef HYDRAMA2D_OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

} // namespace hydrama
