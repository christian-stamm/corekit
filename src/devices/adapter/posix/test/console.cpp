#include "corekit/console.hpp"

#include <gtest/gtest.h>

namespace corekit {

    TEST(Console, WriteSingleByte) {
        Console console("TestConsole");

        ASSERT_TRUE(console.load());
        ASSERT_TRUE(console.isLoaded());

        uint8_t data = 42;
        EXPECT_TRUE(console.write(data));
    }

}  // namespace corekit
