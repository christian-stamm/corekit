#pragma once
#include <atomic>
#include <memory>

namespace corekit {

    template <typename T>
    class PosixAtomic : public std::atomic<T> {
       public:
        using Ptr       = std::shared_ptr<PosixAtomic<T>>;
        using ValueType = T;
        using std::atomic<T>::atomic;

        
    };

}  // namespace corekit