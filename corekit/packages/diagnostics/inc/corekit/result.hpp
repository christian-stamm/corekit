#pragma once

#include <expected>
#include <optional>

#include "corekit/error.hpp"

namespace corekit {

    template <typename T>
    struct Result : public std::expected<T, Error> {
        Result(const T& value = T()) : std::expected<T, Error>(value) {}

        Result(const Error& error)
            : std::expected<T, Error>(std::unexpected(error)) {}
    };

    template <>
    struct Result<void> : public std::expected<void, Error> {
        Result() : std::expected<void, Error>(std::in_place) {}

        Result(const Error& error)
            : std::expected<void, Error>(std::unexpected(error)) {}
    };

    using VoidResult = Result<void>;
    using BoolResult = Result<bool>;

};  // namespace corekit