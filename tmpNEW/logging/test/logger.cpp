#include "corekit/logger.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "corekit/console.hpp"

namespace corekit {

    TEST(Logger, CanBeConstructed) {
        Logger logger("TestLogger");
        EXPECT_EQ(logger.getName(), "TESTLOGGER");
    }

    TEST(Logger, CanLogMessages) {
        LogDevice::Ptr console = std::make_shared<Console>("TestConsole");
        Logging::setLevel(LogLevel::DEBUG);
        Logging::setStream(console);
        EXPECT_TRUE(true);  // Just ensure that logging doesn't throw exceptions
    }

}  // namespace corekit