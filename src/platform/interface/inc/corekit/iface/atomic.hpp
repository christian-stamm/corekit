#pragma once

namespace corekit {

    template <typename T>
    class IAtomic {
       public:
        virtual T    load() const                             = 0;
        virtual void store(T value)                           = 0;
        virtual bool compare_exchange(T& expected, T desired) = 0;
    };

};  // namespace corekit