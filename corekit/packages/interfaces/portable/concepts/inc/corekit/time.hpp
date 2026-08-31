#pragma once

#include "corekit/platform/time.hpp"

namespace corekit {

    using Time = platform::Time;

    template <typename T>
    concept Timeable = requires {
        { T::uptime() } -> std::convertible_to<double>;
        { T::sleep(0.0f) } -> std::same_as<void>;
    };

}  // namespace corekit