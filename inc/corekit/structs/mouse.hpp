#pragma once

#include "corekit/structs/vector.hpp"

namespace corekit {
    namespace structs {

        struct Mouse {
            Vec2 pos         = {};
            Vec2 wheel       = {};
            bool leftButton  = false;
            bool rightButton = false;
        };

    };  // namespace structs
};  // namespace corekit
