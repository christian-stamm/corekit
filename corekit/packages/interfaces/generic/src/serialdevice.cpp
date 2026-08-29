#include "corekit/serialdevice.hpp"

#include <iostream>
namespace corekit {

    template <typename T>
    bool SerialDevice<T>::write_bulk(std::span<const T> data) {
        for (const T& item : data) {
            if (!write(item)) {
                return false;
            }
        }

        return true;
    }

    template <typename T>
    bool SerialDevice<T>::read_bulk(std::span<T> data) {
        for (T& item : data) {
            if (!read(item)) {
                return false;
            }
        }

        return true;
    }

    template <typename T>
    bool SerialDevice<T>::xfer(const T& txData, T& rxData) {
        const bool writeSuccess = write(txData);
        const bool readSuccess  = read(rxData);
        return writeSuccess && readSuccess;
    }

    template <typename T>
    bool SerialDevice<T>::xferBulk(std::span<const T> txData,
                                   std::span<T>       rxData) {
        const uint num_cycles = txData.size();

        if (rxData.size() < num_cycles) {
            return false;
        }

        for (size_t cycle = 0; cycle < num_cycles; ++cycle) {
            if (!xfer(txData[cycle], rxData[cycle])) {
                return false;
            }
        }
        return true;
    }

    template class SerialDevice<uint8_t>;
    template class SerialDevice<uint16_t>;
    template class SerialDevice<uint32_t>;

};  // namespace corekit
