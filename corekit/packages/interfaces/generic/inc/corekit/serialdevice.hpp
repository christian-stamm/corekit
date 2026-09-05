#pragma once
#include <memory>
#include <span>
#include <type_traits>

#include "corekit/basedevice.hpp"

namespace corekit {

    template <typename T>
    class SerialDevice : public BaseDevice {
        static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>,
                      "SerialDevice only supports uint8_t, uint16_t, uint32_t, "
                      "and uint64_t types");

       public:
        using Ptr = std::shared_ptr<SerialDevice<T>>;
        using BaseDevice::BaseDevice;

        virtual bool write(const T& data) = 0;
        virtual bool write_burst(std::span<const T> data);

        virtual bool read(T& data) = 0;
        virtual bool read_burst(std::span<T> data);

        virtual bool xfer(const T& txData, T& rxData);
        virtual bool xferBulk(std::span<const T> txData, std::span<T> rxData);
    };

    extern template class SerialDevice<uint8_t>;
    extern template class SerialDevice<uint16_t>;
    extern template class SerialDevice<uint32_t>;
    extern template class SerialDevice<uint64_t>;

};  // namespace corekit
