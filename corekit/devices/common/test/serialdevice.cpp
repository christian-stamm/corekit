#include "corekit/serialdevice.hpp"

#include <gtest/gtest.h>

#include <sstream>

namespace corekit {

    class SerialTestDev : public SerialDevice<uint32_t> {
       public:
        using SerialDevice<uint32_t>::SerialDevice;

        virtual bool write(const uint32_t& data) override {
            buffer << data;
            return true;
        }

        virtual bool read(uint32_t& data) override {
            buffer >> data;
            return true;
        }

       private:
        std::stringstream buffer;
    };

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    TEST(SerialDevice, NotImplemented) {
        SerialTestDev device("TestDevice");
        EXPECT_FALSE(device.isLoaded());
    }

}  // namespace corekit