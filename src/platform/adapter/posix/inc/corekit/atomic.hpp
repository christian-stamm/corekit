#pragma once
#include <atomic>
#include <memory>

#include "corekit/iface/atomic.hpp"

namespace corekit {

    template <typename T>
    class AtomicPosix : public IAtomic<T> {
       public:
        using Ptr = std::shared_ptr<AtomicPosix<T>>;

        AtomicPosix(T value = T()) : value(value) {}

        virtual T load() const override {
            return value.load();
        }

        virtual void store(T value) override {
            this->value.store(value);
        }

        virtual bool compare_exchange(T& expected, T desired) override {
            return value.compare_exchange_strong(expected, desired);
        }

       private:
        std::atomic<T> value;
    };

    template <typename T>
    using Atomic = AtomicPosix<T>;

}  // namespace corekit