#pragma once
#include <atomic>
#include <memory>

namespace corekit::platform {

    template <typename T>
    class Atomic : public std::atomic<T> {
        using Ptr = std::shared_ptr<Atomic<T>>;
        using std::atomic<T>::atomic;

       public:
        bool compare_exchange(T& expected, T desired) {
            return std::atomic<T>::compare_exchange_strong(expected, desired);
        }
    };

};  // namespace corekit::platform