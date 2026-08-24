#include "corekit/serialdevice.hpp"

#include <gtest/gtest.h>

#include <sstream>

extern "C" {

void vApplicationMallocFailedHook(void) {
    FAIL() << "Malloc failed";
}

void vApplicationStackOverflowHook(TaskHandle_t xTask, char* pcTaskName) {
    FAIL() << "Stack overflow in task: " << pcTaskName;
}

void vApplicationDaemonTaskStartupHook(void) {
    // This function is called when the FreeRTOS daemon task starts up.
    // You can perform any necessary initialization here.
}

}  // extern "C"

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