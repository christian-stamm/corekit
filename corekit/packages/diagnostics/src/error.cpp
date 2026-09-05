#include "corekit/error.hpp"

#include <format>

namespace corekit {

    Error::Stack Error::stack;

    Error::Location::Location(const std::source_location& location)
        : file(location.file_name())
        , func(location.function_name())
        , line(location.line())
        , column(location.column()) {}

    Error::Error(  //
        uint16_t             code,
        std::string          type,
        std::string          message,
        std::source_location location  //
        )
        : code(code)
        , type(type)
        , message(message)
        , location(location) {}

    std::string Error::what() const {
        return std::format(  //
            "\n{}:\n"
            "\tFile: {}\n"
            "\tFunc: {}\n"
            "\tLine: {} ({})\n"
            "\tDesc: {}",
            type,
            location.file,
            location.func,
            location.line,
            location.column,
            message  //
        );
    }

    RuntimeError::RuntimeError(std::string          message,
                               std::source_location location)
        : Error(CODE,
                "RUNTIME_ERROR",
                std::move(message),
                std::move(location)) {}

    NotImplementedError::NotImplementedError(std::string          message,
                                             std::source_location location)
        : Error(CODE,
                "NOT_IMPLEMENTED_ERROR",
                std::move(message),
                std::move(location)) {}

    InvalidArgumentError::InvalidArgumentError(std::string          message,
                                               std::source_location location)
        : Error(CODE,
                "INVALID_ARGUMENT_ERROR",
                std::move(message),
                std::move(location)) {}

    OutOfRangeError::OutOfRangeError(std::string          message,
                                     std::source_location location)
        : Error(CODE,
                "OUT_OF_RANGE_ERROR",
                std::move(message),
                std::move(location)) {}

    TimeoutError::TimeoutError(std::string          message,
                               std::source_location location)
        : Error(CODE,
                "TIMEOUT_ERROR",
                std::move(message),
                std::move(location)) {}

}  // namespace corekit