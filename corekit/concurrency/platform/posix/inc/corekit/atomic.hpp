#pragma once
#include <atomic>
#include <memory>

namespace corekit {

    template <typename T>
    class PosixAtomic {
       public:
        using Ptr       = std::shared_ptr<PosixAtomic<T>>;
        using ValueType = T;

        PosixAtomic(T value = T()) : value(value) {}

        T load() const {
            return value.load();
        }

        void store(T value) {
            this->value.store(value);
        }

        bool compare_exchange(T& expected, T desired) {
            return value.compare_exchange_strong(expected, desired);
        }

       private:
        std::atomic<T> value;
    };

}  // namespace corekit