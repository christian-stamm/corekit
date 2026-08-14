#pragma once
#include <memory>

namespace corekit {

    template <typename T>
    class PicoAtomic {
       public:
        using Ptr       = std::shared_ptr<PicoAtomic<T>>;
        using ValueType = T;

        PicoAtomic(T value = T()) : value(value) {}

        T load() const {
            return value;
        }

        void store(T value) {
            this->value = value;
        }

        bool compare_exchange(T& expected, T desired) {
            if (value == expected) {
                value = desired;
                return true;
            } else {
                expected = value;
                return false;
            }
        }

       private:
        T value;
    };

}  // namespace corekit