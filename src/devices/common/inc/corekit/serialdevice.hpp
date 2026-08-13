#pragma once
#include <memory>

#include "corekit/basedevice.hpp"

namespace corekit {

    template <typename T>
    class SerialDevice : public BaseDevice {
        static_assert(
            std::is_same_v<T, uint8_t>           //
                || std::is_same_v<T, uint16_t>   //
                || std::is_same_v<T, uint32_t>,  //
            "SerialDevice only supports uint8_t, uint16_t, and uint32_t types");

       public:
        using Ptr = std::shared_ptr<SerialDevice<T>>;
        using BaseDevice::BaseDevice;

        virtual bool write(const T& data) = 0;
        virtual bool writeBulk(std::span<const T> data);

        virtual bool read(T& data) = 0;
        virtual bool readBulk(std::span<T> data);
    };

};  // namespace corekit
