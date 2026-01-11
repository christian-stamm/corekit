#pragma once

#include "corekit/utils/math.hpp"

namespace corekit {
    namespace device {

        using namespace corekit::math;

        struct Mouse {
            Vec2 pos         = {};
            Vec2 wheel       = {};
            bool leftButton  = false;
            bool rightButton = false;
        };

    };  // namespace device
};  // namespace corekit
