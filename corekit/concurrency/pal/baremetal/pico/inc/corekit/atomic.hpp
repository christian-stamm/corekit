#pragma once

#include <memory>
#include <mutex>

#include "corekit/mutex.hpp"

namespace corekit {

    template <typename T>
    class PicoAtomic {
       public:
        using Ptr       = std::shared_ptr<PicoAtomic<T>>;
        using ValueType = T;

        PicoAtomic(T value = T()) : value(value) {}

        PicoAtomic(const PicoAtomic&) = delete;
        PicoAtomic(PicoAtomic&&)      = delete;

        PicoAtomic& operator=(const PicoAtomic&) = delete;
        PicoAtomic& operator=(PicoAtomic&&)      = delete;

        T load() const {
            std::lock_guard lock(m_mutex);
            return value;
        }

        void store(T desired) {
            std::lock_guard lock(m_mutex);
            value = desired;
        }

        bool compare_exchange_strong(T& expected, T desired) {
            std::lock_guard lock(m_mutex);

            if (value == expected) {
                value = desired;
                return true;
            }

            expected = value;
            return false;
        }

       private:
        mutable Mutex m_mutex;
        T             value;
    };

    template <typename T>
    using Atomic = PicoAtomic<T>;

}  // namespace corekit