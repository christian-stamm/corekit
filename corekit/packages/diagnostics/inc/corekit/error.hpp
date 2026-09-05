#pragma once

#include <deque>
#include <iostream>
#include <mutex>
#include <source_location>
#include <string>

#include "corekit/mutex.hpp"

namespace corekit {

    constexpr uint16_t MAX_STACK_SIZE = 8;

    struct Error {
       public:
        struct Location {
            Location(const std::source_location& location);

            std::string file;
            std::string func;
            int         line;
            int         column;
        };

        struct Stack {
            const Error* top() const {
                std::lock_guard lock(m_mutex);
                if (m_stack.empty()) {
                    return nullptr;
                }
                return &m_stack.back();
            }

            void pop() {
                std::lock_guard lock(m_mutex);
                m_stack.pop_back();
            }

            bool push(const Error& error) {
                std::lock_guard lock(m_mutex);
                if (MAX_STACK_SIZE <= m_stack.size()) {
                    return false;
                }

                m_stack.push_back(error);
                return true;
            }

            bool empty() const {
                std::lock_guard lock(m_mutex);
                return m_stack.empty();
            }

            void clear() {
                std::lock_guard lock(m_mutex);
                m_stack.clear();
            }

            void dump(std::ostream& os) const {
                std::lock_guard lock(m_mutex);
                for (const auto& error : m_stack) {
                    os << error.what() << std::endl;
                }
            }

           private:
            std::deque<Error> m_stack;
            mutable Mutex     m_mutex;
        };

        explicit Error(
            uint16_t             code     = 0,
            std::string          type     = "",
            std::string          message  = "",
            std::source_location location = std::source_location::current());

        std::string what() const;

        uint16_t    code;
        std::string type;
        std::string message;
        Location    location;

        friend std::ostream& operator<<(std::ostream& os, const Error& error) {
            os << error.what();
            return os;
        }

        static Stack stack;
    };

    class RuntimeError : public Error {
       public:
        constexpr static uint16_t CODE = 1;
        explicit RuntimeError(
            std::string          message  = "",
            std::source_location location = std::source_location::current());
    };

    class NotImplementedError : public Error {
       public:
        constexpr static uint16_t CODE = 2;
        explicit NotImplementedError(
            std::string          message  = "",
            std::source_location location = std::source_location::current());
    };

    class InvalidArgumentError : public Error {
       public:
        constexpr static uint16_t CODE = 3;
        explicit InvalidArgumentError(
            std::string          message  = "",
            std::source_location location = std::source_location::current());
    };

    class OutOfRangeError : public Error {
       public:
        constexpr static uint16_t CODE = 4;
        explicit OutOfRangeError(
            std::string          message  = "",
            std::source_location location = std::source_location::current());
    };

    class TimeoutError : public Error {
       public:
        constexpr static uint16_t CODE = 5;
        explicit TimeoutError(
            std::string          message  = "",
            std::source_location location = std::source_location::current());
    };

}  // namespace corekit