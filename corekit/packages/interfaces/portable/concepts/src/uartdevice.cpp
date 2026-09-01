#include "corekit/uartdevice.hpp"

#include "corekit/serialdevice.hpp"

namespace corekit {

    static_assert(std::is_base_of_v<SerialDevice<uint8_t>, Uart::Device>,
                  "UartDevice must be a subclass of SerialDevice");

}  // namespace corekit