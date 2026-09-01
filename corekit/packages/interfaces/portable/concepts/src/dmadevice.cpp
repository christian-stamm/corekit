#include "corekit/dmadevice.hpp"

#include "corekit/serialdevice.hpp"

namespace corekit {

    static_assert(std::is_base_of_v<SerialDevice<uint32_t>, DmaDevice>,
                  "DmaDevice must be a subclass of SerialDevice");

}  // namespace corekit