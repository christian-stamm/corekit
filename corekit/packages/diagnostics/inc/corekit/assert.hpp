#pragma once

#include <expected>
#include <optional>

#include "corekit/error.hpp"
#include "corekit/result.hpp"

namespace corekit {

    extern VoidResult corecheck(bool         condition,
                                const Error& error = RuntimeError());

};  // namespace corekit