#pragma once

#include "corekit/atomic.hpp"
#include "corekit/mutex.hpp"
#include "corekit/smphr.hpp"
#include "corekit/thread.hpp"

namespace corekit {

    template <typename T>
    using Atomic    = PicoAtomic<T>;
    using Mutex     = PicoMutex;
    using Semaphore = PicoSemaphore;
    using Thread    = PicoThread;

}  // namespace corekit