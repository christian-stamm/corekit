#pragma once
#include <cstddef>

namespace corekit::mem {

    extern void* malloc(size_t size, size_t alignment = 0);
    extern void  free(void* ptr);

} // namespace corekit::mem