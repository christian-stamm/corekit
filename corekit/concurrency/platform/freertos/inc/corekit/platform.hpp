#pragma once

#include "corekit/atomic.hpp"
#include "corekit/mutex.hpp"
#include "corekit/smphr.hpp"
#include "corekit/thread.hpp"

namespace corekit {

    template <typename T>
    using Atomic    = FreeRTOSAtomic<T>;
    using Mutex     = FreeRTOSMutex;
    using Semaphore = FreeRTOSSemaphore;
    using Thread    = FreeRTOSThread;

}  // namespace corekit