#include "corekit/console.hpp"

#include <gtest/gtest.h>

#include <memory>

namespace corekit {

    TEST(Console, WriteSingleByte) {
        Console::Ptr console = std::make_shared<Console>("TestConsole");

        ASSERT_TRUE(console->load());
        ASSERT_TRUE(console->isLoaded());
    }

}  // namespace corekit
