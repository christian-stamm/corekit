#pragma once

#include <iostream>

#include "corekit/serialdevice.hpp"

namespace corekit {

    using StreamDevice = SerialDevice<uint8_t>;

    class StreamBuffer : public std::streambuf {
       public:
        using Ptr = std::shared_ptr<StreamBuffer>;
        StreamBuffer(const StreamDevice::Ptr& device);

        virtual std::streambuf::int_type overflow(
            std::streambuf::int_type c) override;
        virtual std::streamsize xsputn(const char*     s,
                                       std::streamsize count) override;

        virtual std::streambuf::int_type underflow() override;
        virtual std::streamsize xsgetn(char* s, std::streamsize count) override;

       private:
        char              buffer[1]{0};
        StreamDevice::Ptr device;
    };

};  // namespace corekit
