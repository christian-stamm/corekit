#pragma once

#include <memory>

#include "corekit/platform/piodevice.hpp"

namespace corekit {

    template <typename T>
    using PioDevice = platform::Pio::Node<T>;

}  // namespace corekit