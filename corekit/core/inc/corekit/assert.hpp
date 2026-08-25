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
            NONE             = 0,
            RUNTIME_ERROR    = 1,
            NOT_IMPLEMENTED  = 2,
            INVALID_ARGUMENT = 3,
            OUT_OF_RANGE     = 4,
            TIMEOUT          = 5,
        };

        Error(Type type = Type::NONE, const Message& message = "<NO ERROR>")
            : type(type)
            , message(message) {}

        bool operator==(const Error& other) const {
            return type == other.type;
        }

        operator bool() const {
            return type != Type::NONE;
        }

        Message trace(const Location& location = Location::current()) const {
            return std::format("File: {}\nFunc: {}\nLine: {} ({})\nDesc: {}",
                               location.file_name(),
                               location.function_name(),
                               location.line(),
                               location.column(),
                               message);
        }

        Type     type;
        Message  message;
        Location location;
    };

    class RuntimeError : public Error {
       public:
        RuntimeError(const Message& message = "")
            : Error(Error::Type::RUNTIME_ERROR, message) {}
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

    template <typename T>
    using Result = std::expected<T, Error>;

    extern Result<void> corecheck(bool         condition,
                                  const Error& error = RuntimeError());

};  // namespace corekit