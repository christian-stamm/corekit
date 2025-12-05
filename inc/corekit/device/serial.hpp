#pragma once

#include "corekit/device/device.hpp"

#include <span>

namespace corekit {
namespace device {

    template <typename T> class Serial : public Device {
      public:
        using Ptr = std::shared_ptr<Serial<T>>;
        using Device::Device;

        virtual bool read(std::span<T>& dst) const        = 0;
        virtual bool write(const std::span<T>& src) const = 0;
        virtual bool xfer(const std::span<T>& src, std::span<T>& dst) const;
    };

}; // namespace device
}; // namespace corekit
