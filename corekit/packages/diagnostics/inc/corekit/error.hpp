#pragma once

#include <iostream>
#include <source_location>
#include <string>

namespace corekit {

    struct Error {
        public:

            struct Location {
                    Location(const std::source_location& location);

                    std::string file;
                    std::string func;
                    int         line;
                    int         column;
            };

            explicit Error(uint16_t code = 0, std::string type = "", std::string message = "", std::source_location location = std::source_location::current());

            std::string what() const;

            uint16_t    code;
            std::string type;
            std::string message;
            Location    location;

            friend std::ostream& operator<<(std::ostream& os, const Error& error) {
                os << error.what();
                return os;
            }
    };

    class RuntimeError : public Error {
        public:

            constexpr static uint16_t CODE = 1;
            explicit RuntimeError(std::string message = "", std::source_location location = std::source_location::current());
    };

    class NotImplementedError : public Error {
        public:

            constexpr static uint16_t CODE = 2;
            explicit NotImplementedError(std::string message = "", std::source_location location = std::source_location::current());
    };

    class InvalidArgumentError : public Error {
        public:

            constexpr static uint16_t CODE = 3;
            explicit InvalidArgumentError(std::string message = "", std::source_location location = std::source_location::current());
    };

    class OutOfRangeError : public Error {
        public:

            constexpr static uint16_t CODE = 4;
            explicit OutOfRangeError(std::string message = "", std::source_location location = std::source_location::current());
    };

    class TimeoutError : public Error {
        public:

            constexpr static uint16_t CODE = 5;
            explicit TimeoutError(std::string message = "", std::source_location location = std::source_location::current());
    };

} // namespace corekit