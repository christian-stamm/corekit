#include "corekit/memory.hpp"

namespace corekit::mem {

    void* malloc(size_t size, size_t alignment) {
        if (0 < alignment) {
            return std::aligned_alloc(alignment, size);
        } else {
            return std::malloc(size);
        }
    }

    void free(void* ptr) { std::free(ptr); }

} // namespace corekit::mem