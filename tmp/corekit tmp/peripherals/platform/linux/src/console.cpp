#include "corekit/console.hpp"

#include <iostream>

namespace corekit {

    bool Console::write(const uint8_t& data) {
        std::cout.put(static_cast<char>(data));
        return static_cast<bool>(std::cout);
    }

    bool Console::writeBulk(std::span<const uint8_t> data) {
        if (data.empty())
            return true;

        const char* buffer = reinterpret_cast<const char*>(data.data());
        std::cout.write(buffer, data.size());

        return static_cast<bool>(std::cout);
    }

    bool Console::read(uint8_t& data) {
        char c;

        if (!std::cin.get(c))
            return false;

        data = static_cast<uint8_t>(c);

        return true;
    }

    bool Console::readBulk(std::span<uint8_t> data) {
        if (data.empty())
            return true;

        std::cin.read(reinterpret_cast<char*>(data.data()),
                      static_cast<std::streamsize>(data.size()));

        return std::cin.gcount() == static_cast<std::streamsize>(data.size());
    }

}  // namespace corekit
