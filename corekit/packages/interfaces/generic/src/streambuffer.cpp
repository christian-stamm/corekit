#include "corekit/streambuffer.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace corekit {

    StreamBuffer::StreamBuffer(const StreamDevice::Ptr& device)
        : device(device) {}

    std::streambuf::int_type StreamBuffer::overflow(
        std::streambuf::int_type c) {
        if (c != traits_type::eof() && device != nullptr) {
            device->write(static_cast<uint8_t>(c));
            return std::streambuf::traits_type::to_int_type(c);
        }

        return traits_type::eof();
    }

    std::streambuf::int_type StreamBuffer::underflow() {
        if (device == nullptr)
            return traits_type::eof();

        uint8_t data;

        if (!device->read(data))
            return traits_type::eof();

        buffer[0] = static_cast<char>(data);
        setg(buffer, buffer, buffer + 1);

        return traits_type::to_int_type(buffer[0]);
    }

    std::streamsize StreamBuffer::xsputn(const char* s, std::streamsize count) {
        if (0 < count && device != nullptr) {
            if (device->write_burst(std::span<const uint8_t>(
                    reinterpret_cast<const uint8_t*>(s),
                    static_cast<std::size_t>(count)))) {
                return count;
            }
        }

        return 0;
    }

    std::streamsize StreamBuffer::xsgetn(char* s, std::streamsize count) {
        if (count <= 0 || !device)
            return 0;

        uint8_t*    ptr  = reinterpret_cast<uint8_t*>(s);
        std::size_t size = static_cast<std::size_t>(count);

        auto buffer = std::span<uint8_t>(ptr, size);

        return device->read_burst(buffer) ? count : 0;
    }

}  // namespace corekit