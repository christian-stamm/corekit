#include "corekit/error.hpp"

#include <format>

namespace corekit {

    Error::Error(Type type, const Message& message, const Location& location)
        : type(type)
        , message(message)
        , location(location) {}

    Error::operator bool() const {
        return type != Type::NONE;
    }

    Error::Message Error::traceback() const {
        return std::format(
            "{} ERROR\n\n"
            "File: {}\n"
            "Func: {}\n"
            "Line: {} ({})\n"
            "Desc: {}",
            type_to_string(),
            location.file_name(),
            location.function_name(),
            location.line(),
            location.column(),
            message);
    }

    Error::Message Error::what() const {
        return message;
    }

    Error::Message Error::type_to_string() const {
        switch (type) {
            case Type::NONE: return "NONE";
            case Type::RUNTIME: return "RUNTIME";
            case Type::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
            case Type::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
            case Type::OUT_OF_RANGE: return "OUT_OF_RANGE";
            case Type::TIMEOUT: return "TIMEOUT";
            default: return "UNDEFINED";
        }
    }

    RuntimeError::RuntimeError(const Message& message)
        : Error(Type::RUNTIME, message) {}

    NotImplementedError::NotImplementedError(const Message& message)
        : Error(Type::NOT_IMPLEMENTED, message) {}

    InvalidArgumentError::InvalidArgumentError(const Message& message)
        : Error(Type::INVALID_ARGUMENT, message) {}

    OutOfRangeError::OutOfRangeError(const Message& message)
        : Error(Type::OUT_OF_RANGE, message) {}

    TimeoutError::TimeoutError(const Message& message)
        : Error(Type::TIMEOUT, message) {}

}  // namespace corekit