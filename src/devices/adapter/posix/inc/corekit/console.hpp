#pragma once

#include "corekit/serialdevice.hpp"

namespace corekit {

    class Console : public SerialDevice<uint8_t> {
       public:
        using Ptr = std::shared_ptr<Console>;
        using SerialDevice<uint8_t>::SerialDevice;

        virtual bool write(const uint8_t& data) override;
        virtual bool writeBulk(const std::span<uint8_t>& data) override;
        virtual bool read(uint8_t& data) override;
        virtual bool readBulk(std::span<uint8_t>& data) override;
    };

}  // namespace corekit