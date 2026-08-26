#pragma once

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

        explicit Error(Type            type     = Type::NONE,
                       const Message&  message  = "<NO ERROR>",
                       const Location& location = Location::current());

        operator bool() const;

        Message traceback() const;
        Message what() const;

       private:
        Message type_to_string() const;

       public:
        Type     type;
        Message  message;
        Location location;
    };

    class RuntimeError : public Error {
       public:
        explicit RuntimeError(const Message& message = "");
    };

    class NotImplementedError : public Error {
       public:
        explicit NotImplementedError(const Message& message = "");
    };

    class InvalidArgumentError : public Error {
       public:
        explicit InvalidArgumentError(const Message& message = "");
    };

    class OutOfRangeError : public Error {
       public:
        explicit OutOfRangeError(const Message& message = "");
    };

    class TimeoutError : public Error {
       public:
        explicit TimeoutError(const Message& message = "");
    };

}  // namespace corekit