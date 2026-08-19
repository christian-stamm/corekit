#pragma once
#include <atomic>
#include <memory>

namespace corekit {

    template <typename T>
    class StdlibAtomic : public std::atomic<T> {
       public:
        using Ptr = std::shared_ptr<StdlibAtomic<T>>;
        using std::atomic<T>::atomic;
    };

    template <typename T>
    using Atomic = StdlibAtomic<T>;

}  // namespace corekit