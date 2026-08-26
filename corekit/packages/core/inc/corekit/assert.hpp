#pragma once

#include <expected>

#include "corekit/error.hpp"

namespace corekit {

    template <typename T>
    using Result = std::expected<T, Error>;

    extern Result<void> corecheck(bool         condition,
                                  const Error& error = RuntimeError());

};  // namespace corekit