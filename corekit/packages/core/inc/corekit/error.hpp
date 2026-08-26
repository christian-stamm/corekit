#pragma once

#include <expected>
#include <format>
#include <source_location>
#include <string>

namespace corekit {

    using Location = std::source_location;

    struct Error {
        using Message = std::string;

        enum class Type {
            NONE,
            RUNTIME,
            NOT_IMPLEMENTED,
            INVALID_ARGUMENT,
            OUT_OF_RANGE,
            TIMEOUT,
        };

        Error(Type            type     = Type::NONE,
              const Message&  message  = "<NO ERROR>",
              const Location& location = Location::current())
            : type(type)
            , message(message)
            , location(location) {}

        operator bool() const {
            return type != Type::NONE;
        }

        Message trace() const {
            return std::format(
                "{} ERROR\n\nFile: {}\nFunc: {}\nLine: {} ({})\nDesc: {}",
                type_to_string(),
                location.file_name(),
                location.function_name(),
                location.line(),
                location.column(),
                message);
        }

       private:
        Message type_to_string() const {
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

        Type     type;
        Message  message;
        Location location;
    };

    class RuntimeError : public Error {
       public:
        RuntimeError(const Message& message = "")
            : Error(Error::Type::RUNTIME, message) {}
    };

    class NotImplementedError : public Error {
       public:
        NotImplementedError(const Message& message = "")
            : Error(Error::Type::NOT_IMPLEMENTED, message) {}
    };

    class InvalidArgumentError : public Error {
       public:
        InvalidArgumentError(const Message& message = "")
            : Error(Error::Type::INVALID_ARGUMENT, message) {}
    };

    class OutOfRangeError : public Error {
       public:
        OutOfRangeError(const Message& message = "")
            : Error(Error::Type::OUT_OF_RANGE, message) {}
    };

    class TimeoutError : public Error {
       public:
        TimeoutError(const Message& message = "")
            : Error(Error::Type::TIMEOUT, message) {}
    };

};  // namespace corekit