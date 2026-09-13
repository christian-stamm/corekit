#include "corekit/serialdevice.hpp"

#include "corekit/check.hpp"

namespace corekit {

    template <typename T>
    bool SerialDevice<T>::write_burst(std::span<const T> data) {
        core::check(is_loaded(), RuntimeError("Serial device is not loaded"));
        bool success = core::ok();
        for (const T& item : data) { success &= write(item); }
        return core::check(success, RuntimeError("Failed to write burst data"));
    }

    template <typename T>
    bool SerialDevice<T>::read_burst(std::span<T> data) {
        core::check(is_loaded(), RuntimeError("Serial device is not loaded"));
        bool success = core::ok();
        for (T& item : data) { success &= read(item); }
        return core::check(success, RuntimeError("Failed to read burst data"));
    }

    template <typename T>
    bool SerialDevice<T>::xfer(const T& txData, T& rxData) {
        core::check(is_loaded(), RuntimeError("Serial device is not loaded"));
        core::check(write(txData), RuntimeError("Failed to write data"));
        core::check(read(rxData), RuntimeError("Failed to read data"));
        return core::ok();
    }

    template <typename T>
    bool SerialDevice<T>::xfer_burst(std::span<const T> txData, std::span<T> rxData) {
        core::check(is_loaded(), RuntimeError("Serial device is not loaded"));
        core::check(txData.size() <= rxData.size(), RuntimeError("rxData buffer is too small"));
        bool success = core::ok();
        for (size_t cycle = 0; cycle < txData.size(); ++cycle) { success &= xfer(txData[cycle], rxData[cycle]); }
        return core::check(success, RuntimeError("Failed to perform bulk transfer"));
    }

    template class SerialDevice<uint8_t>;
    template class SerialDevice<uint16_t>;
    template class SerialDevice<uint32_t>;
    template class SerialDevice<uint64_t>;

}; // namespace corekit
