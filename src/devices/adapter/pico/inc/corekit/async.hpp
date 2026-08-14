#pragma once

#include <hardware/uart.h>

#include "corekit/serialdevice.hpp"

namespace corekit {

    struct CtrlBlock {
        const volatile void* addr;
        const uint           dreq;
    };

    template <typename T>
    class AsyncDevice : public SerialDevice<T> {
       public:
        using Ptr = std::shared_ptr<AsyncDevice>;

        AsyncDevice(const Name&      name,
                    const CtrlBlock& reader,
                    const CtrlBlock& writer)
            : SerialDevice<T>(name)
            , reader(reader)
            , writer(writer){};

        const CtrlBlock reader;
        const CtrlBlock writer;
    };

}  // namespace corekit