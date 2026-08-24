#include "corekit/assert.hpp"

#include <format>

namespace corekit {

    void corecheck(bool                        condition,
                   const std::string&          message,
                   const std::source_location& location) {
#ifndef NDEBUG
        if (!condition) {
            throw std::runtime_error(
                std::format("Assertion failed:\n\n\tFile: {}\n\tFunc: "
                            "{}\n\tLine: {} ({})\n\tDesc: {}\n\n"
                            "\tStack:\n{}",
                            location.file_name(),
                            location.function_name(),
                            location.line(),
                            location.column(),
                            message,
                            trace));
        }
#endif
    }
};  // namespace corekit
