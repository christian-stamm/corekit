#pragma once

#include <source_location>
#include <string>

#include "corekit/utils/assert.hpp"

// KEEP GLAD BEFORE GLFW
#include <glad/glad.h>
// KEEP GLAD BEFORE GLFW
#include <GLFW/glfw3.h>
// KEEP GLAD BEFORE GLFW

namespace corekit {
    namespace opengl {

        using namespace corekit::utils;
        using namespace corekit::types;

        void glCheckError(const Status&   message  = "<NO DESCRIPTION>",
                          const Location& location = Location::current());

    };  // namespace opengl
};      // namespace corekit
