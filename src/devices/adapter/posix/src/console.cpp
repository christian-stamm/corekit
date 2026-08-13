#include "corekit/console.hpp"

#include <iostream>

namespace corekit {

    bool Console::write(const uint8_t& data) {
        std::cout << data << std::flush;
        return true;
    }

    bool Console::writeBulk(const std::span<uint8_t>& data) {
        const char* buffer = reinterpret_cast<const char*>(data.data());
        std::cout.write(buffer, data.size());
        std::cout << std::flush;
        return true;
    }

    bool Console::read(uint8_t& data) {
        std::cin >> data;
        return true;
    }

    bool Console::readBulk(std::span<uint8_t>& data) {
        char* buffer = reinterpret_cast<char*>(data.data());
        std::cin.read(buffer, data.size());
        return true;
    }

}  // namespace corekit
