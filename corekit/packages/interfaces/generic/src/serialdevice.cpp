#include "corekit/serialdevice.hpp"

#include <iostream>
namespace corekit {

    template <typename T>
    VoidResult SerialDevice<T>::write_burst(std::span<const T> data) {
        for (const T& item : data) {
            if (!write(item)) {
                return RuntimeError("Failed to write burst data.");
            }
        }

        return VoidResult();
    }

    template <typename T>
    VoidResult SerialDevice<T>::read_burst(std::span<T> data) {
        for (T& item : data) {
            if (!read(item)) {
                return RuntimeError("Failed to read burst data.");
            }
        }

        return VoidResult();
    }

    template <typename T>
    VoidResult SerialDevice<T>::xfer(const T& txData, T& rxData) {
        if (!write(txData) || !read(rxData)) {
            return RuntimeError("Failed to xfer data.");
        }
        return VoidResult();
    }

    template <typename T>
    VoidResult SerialDevice<T>::xfer_burst(std::span<const T> txData,
                                           std::span<T>       rxData) {
        const uint num_cycles = txData.size();

        if (rxData.size() < num_cycles) {
            return RuntimeError("rxData span is smaller than txData span.");
        }

        for (size_t cycle = 0; cycle < num_cycles; ++cycle) {
            if (!xfer(txData[cycle], rxData[cycle])) {
                return RuntimeError("Failed to xfer burst data.");
            }
        }

        return VoidResult();
    }

    template class SerialDevice<uint8_t>;
    template class SerialDevice<uint16_t>;
    template class SerialDevice<uint32_t>;
    template class SerialDevice<uint64_t>;

};  // namespace corekit
