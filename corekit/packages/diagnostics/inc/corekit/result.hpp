#pragma once

#include <expected>
#include <type_traits>

#include "corekit/error.hpp"

namespace corekit {

    template <typename T>
    struct Result : public std::expected<T, Error> {
        using Base = std::expected<T, Error>;

        Result()
            requires std::is_void_v<T>
            : Base(std::in_place) {}

        template <typename U>
            requires(!std::is_void_v<T> && std::constructible_from<T, const U&>)
        Result(const U& value) : Base(std::in_place, value) {}

        template <typename U>
            requires(!std::is_void_v<T> && std::constructible_from<T, U &&>)
        Result(U&& value) : Base(std::forward<U>(value)) {}

        Result(const Error& error) : Base(std::unexpected(error)) {
            Error::stack.push(error);
        }
    };

    using VoidResult = Result<void>;
    using BoolResult = Result<bool>;

}  // namespace corekit