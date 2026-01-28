#include <format>
#include <string>

#include "corekit/core.hpp"

namespace corekit {
    namespace utils {

        void corecheck(bool               condition,
                       const std::string& message,
                       const Location&    location) {
#ifndef NDEBUG
            if (!condition) {
                throw std::runtime_error(
                    std::format("Assertion failed:\n\n\tFile: {}\n\tFunc: "
                                "{}\n\tLine: {} ({})\n\tDesc: {}",
                                location.file_name(),
                                location.function_name(),
                                location.line(),
                                location.column(),
                                message));
            }
#endif
        }

        void glCheckError(std::string meta, const Location& location) {
            const GLenum& errcode = glGetError();

            corecheck(errcode == GL_NO_ERROR,
                      std::format("OpenGL Code: 0x{:04X} ({:08}) {}",
                                  errcode,
                                  errcode,
                                  meta),
                      location);
        }

    };  // namespace utils
};      // namespace corekit
