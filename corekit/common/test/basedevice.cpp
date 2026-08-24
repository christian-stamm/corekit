#include "corekit/basedevice.hpp"

#include <gtest/gtest.h>

extern "C" {

void vApplicationMallocFailedHook(void) {
    FAIL() << "Malloc failed";
}

void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName) {
    FAIL() << "Stack overflow in task: " << pcTaskName;
}

void vApplicationDaemonTaskStartupHook(void) {
    // This function is called when the FreeRTOS daemon task starts up.
    // You can perform any necessary initialization here.
}

}  // extern "C"

namespace corekit {

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    TEST(BaseDevice, StartsUnloaded) {
        BaseDevice device("TestDevice");

        EXPECT_FALSE(device.isLoaded());
    }

    // -------------------------------------------------------------------------
    // Loading
    // -------------------------------------------------------------------------

    TEST(BaseDevice, LoadChangesStateToLoaded) {
        BaseDevice device("TestDevice");

        EXPECT_FALSE(device.isLoaded());
        EXPECT_TRUE(device.load());
        EXPECT_TRUE(device.isLoaded());
    }

    TEST(BaseDevice, LoadingAlreadyLoadedDeviceFails) {
        BaseDevice device("TestDevice");

        ASSERT_TRUE(device.load());
        ASSERT_TRUE(device.isLoaded());
        EXPECT_FALSE(device.load());
        EXPECT_TRUE(device.isLoaded());
    }

    // -------------------------------------------------------------------------
    // Unloading
    // -------------------------------------------------------------------------

    TEST(BaseDevice, UnloadChangesStateToUnloaded) {
        BaseDevice device("TestDevice");

        ASSERT_TRUE(device.load());
        ASSERT_TRUE(device.isLoaded());
        EXPECT_TRUE(device.unload());
        EXPECT_FALSE(device.isLoaded());
    }

    TEST(BaseDevice, UnloadingAlreadyUnloadedDeviceFails) {
        BaseDevice device("TestDevice");

        EXPECT_FALSE(device.isLoaded());
        EXPECT_FALSE(device.unload());
        EXPECT_FALSE(device.isLoaded());
    }

    // -------------------------------------------------------------------------
    // Reloading
    // -------------------------------------------------------------------------

    TEST(BaseDevice, ReloadLoadsUnloadedDevice) {
        BaseDevice device("TestDevice");

        EXPECT_FALSE(device.isLoaded());
        EXPECT_FALSE(device.reload());
        EXPECT_TRUE(device.isLoaded());
    }

    TEST(BaseDevice, ReloadUnloadsAndLoadsLoadedDevice) {
        BaseDevice device("TestDevice");

        ASSERT_TRUE(device.load());
        ASSERT_TRUE(device.isLoaded());
        EXPECT_TRUE(device.reload());
        EXPECT_TRUE(device.isLoaded());
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    TEST(BaseDevice, CompleteLifecycle) {
        BaseDevice device("TestDevice");

        EXPECT_FALSE(device.isLoaded());
        EXPECT_TRUE(device.load());
        EXPECT_TRUE(device.isLoaded());
        EXPECT_TRUE(device.unload());
        EXPECT_FALSE(device.isLoaded());
        EXPECT_FALSE(device.reload());
        EXPECT_TRUE(device.isLoaded());
        EXPECT_TRUE(device.unload());
        EXPECT_FALSE(device.isLoaded());
    }

    TEST(BaseDevice, MultipleLoadUnloadCycles) {
        BaseDevice device("TestDevice");

        for (int i = 0; i < 3; ++i) {
            EXPECT_FALSE(device.isLoaded());

            EXPECT_TRUE(device.load());
            EXPECT_TRUE(device.isLoaded());

            EXPECT_TRUE(device.unload());
            EXPECT_FALSE(device.isLoaded());
        }
    }

}  // namespace corekit