
#include "corekit/opengl/assert.hpp"

#include <format>

#include "corekit/types.hpp"
#include "corekit/utils/assert.hpp"

namespace corekit {
    namespace opengl {

        using namespace corekit::types;
        using namespace corekit::utils;

        void glCheckError(const Status& message, const Location& location) {
            const GLenum& errcode = glGetError();

            corecheck(errcode == GL_NO_ERROR,
                      std::format("OpenGL Code: 0x{:04X} ({:08}) {}",
                                  errcode,
                                  errcode,
                                  message),
                      location);
        }

    };  // namespace opengl
};      // namespace corekit
