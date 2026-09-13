#include "corekit/memory.hpp"

#include <FreeRTOS.h>
#include <task.h>

#include "corekit/check.hpp"

namespace corekit::mem {

    void* malloc(size_t size, size_t alignment) {
        core::check(alignment == 0, NotImplementedError("mem alignment is not supported"));
        return pvPortMalloc(size);
    }

    void free(void* ptr) {
        core::check(ptr != nullptr, RuntimeError("pointer is null"));
        if (core::ok()) { vPortFree(ptr); }
    }

} // namespace corekit::mem