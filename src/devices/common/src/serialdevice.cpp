#include "corekit/serialdevice.hpp"

namespace corekit {

    template <typename T>
    bool SerialDevice<T>::writeBulk(const std::span<T>& data) {
        for (const T& item : data) {
            if (!write(item)) {
                return false;
            }
        }

        return true;
    }

    template <typename T>
    bool SerialDevice<T>::readBulk(std::span<T>& data) {
        for (T& item : data) {
            if (!read(item)) {
                return false;
            }
        }

        return true;
    }

    template class SerialDevice<uint8_t>;
    template class SerialDevice<uint16_t>;
    template class SerialDevice<uint32_t>;

};  // namespace corekit
