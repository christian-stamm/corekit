#include "corekit/device/serial.hpp"

#include <span>

namespace corekit {
    namespace device {

        template <typename T>
        bool Serial<T>::xfer(const std::span<T>& src, std::span<T>& dst) const {
            return write(src) && read(dst);
        }

        template class Serial<uint8_t>;
        template class Serial<uint16_t>;
        template class Serial<uint32_t>;

    };  // namespace device
};  // namespace corekit
