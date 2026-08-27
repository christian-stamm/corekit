#pragma once

#include <memory>
#include <string>

#include "corekit/serialdevice.hpp"

namespace corekit {

    struct CtrlBlock {
        const volatile void* addr;
        const uint32_t       dreq;
    };

    template <typename T>
    class AsyncDevice : public SerialDevice<T> {
       public:
        using Ptr = std::shared_ptr<AsyncDevice>;

        AsyncDevice(const std::string& name,
                    const CtrlBlock&   reader,
                    const CtrlBlock&   writer)
            : SerialDevice<T>(name)
            , reader(reader)
            , writer(writer){};

        const CtrlBlock reader;
        const CtrlBlock writer;
    };

    extern template class AsyncDevice<uint8_t>;
    extern template class AsyncDevice<uint16_t>;
    extern template class AsyncDevice<uint32_t>;

}  // namespace corekit