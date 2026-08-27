#include "corekit/piodevice.hpp"

namespace corekit {

    static_assert(std::is_base_of_v<SerialDevice<uint8_t>, PioDevice<uint8_t>>,
                  "PioDevice must be a subclass of SerialDevice");

}  // namespace corekit