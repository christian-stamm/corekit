#include "corekit/logger.hpp"

#include <gtest/gtest.h>

namespace corekit {

    TEST(Logger, CanBeConstructed) {
        Logger logger("TestLogger");
        EXPECT_EQ(logger.getName(), "TESTLOGGER");
    }

    TEST(Logger, CanLogMessages) {
        Logger    logger("TestLogger");
        LogStream stream = logger(LogLevel::INFO);
        stream << "This is a test log message.";
        EXPECT_TRUE(true);  // Just ensure that logging doesn't throw exceptions
    }

}  // namespace corekit