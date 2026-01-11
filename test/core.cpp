#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "corekit/system/concurrency/flow/executor.hpp"
#include "corekit/system/concurrency/flow/receiver.hpp"
#include "corekit/system/concurrency/flow/scheduler.hpp"
#include "corekit/utils/memory.hpp"
#include "structs.hpp"

// ###################################################################

class DeviceTests : public testing::Test {
   protected:
    void SetUp() override {
        device   = std::make_shared<TestDevice>();
        txBuffer = Memory<uint8_t>(256, 0);
        rxBuffer = Memory<uint8_t>(256, 0);
    }

    void TearDown() override {
        device.reset();
    }

    TestDevice::Ptr device;
    Memory<uint8_t> txBuffer;
    Memory<uint8_t> rxBuffer;
};

TEST_F(DeviceTests, DeviceLoading) {
    EXPECT_FALSE(device->isLoaded());
    EXPECT_TRUE(device->load());
    EXPECT_TRUE(device->isLoaded());
    EXPECT_TRUE(device->unload());
    EXPECT_FALSE(device->isLoaded());
    EXPECT_TRUE(device->reload());
    EXPECT_TRUE(device->isLoaded());
    EXPECT_TRUE(device->reload());
    EXPECT_TRUE(device->isLoaded());
}

TEST_F(DeviceTests, DeviceXfer) {
    ASSERT_TRUE(device->load());

    txBuffer.iota(0);
    rxBuffer.fill(0);
    EXPECT_TRUE(device->write(txBuffer));
    EXPECT_TRUE(device->read(rxBuffer));
    EXPECT_TRUE(txBuffer == rxBuffer);

    txBuffer.fill(42);
    EXPECT_FALSE(txBuffer == rxBuffer);
}

// ###################################################################

TEST(MemoryTests, MemorySplit) {
    Memory<uint32_t> memory(10, 0);
    auto             splits = memory.split(3);

    EXPECT_EQ(splits.size(), 3);
    EXPECT_EQ(splits[0].size(), 3);
    EXPECT_EQ(splits[1].size(), 3);
    EXPECT_EQ(splits[2].size(), 4);
}

TEST(MemoryTests, MemoryInitialization) {
    Memory<uint32_t> memory(5, 42);
    EXPECT_EQ(memory.size(), 5);
    for (size_t i = 0; i < memory.size(); ++i) {
        EXPECT_EQ(memory[i], 42);
    }
}

TEST(MemoryTests, MemoryFill) {
    Memory<uint8_t> memory(10, 0);
    memory.fill(255);
    for (size_t i = 0; i < memory.size(); ++i) {
        EXPECT_EQ(memory[i], 255);
    }
}

TEST(MemoryTests, MemoryIota) {
    Memory<uint32_t> memory(5, 0);
    memory.iota(0);
    for (size_t i = 0; i < memory.size(); ++i) {
        EXPECT_EQ(memory[i], i);
    }
}

TEST(MemoryTests, MemoryComparison) {
    Memory<uint32_t> mem1(5, 0);
    Memory<uint32_t> mem2(5, 0);
    Memory<uint32_t> mem3(5, 1);

    mem1.iota(0);
    mem2.iota(0);
    mem3.iota(0);

    EXPECT_TRUE(mem1 == mem2);
    EXPECT_TRUE(mem1 == mem3);
}

TEST(MemoryTests, MemoryDifferentSizesNotEqual) {
    Memory<uint32_t> mem1(5, 0);
    Memory<uint32_t> mem2(3, 0);

    EXPECT_FALSE(mem1 == mem2);
}

TEST(MemoryTests, MemoryCastMultipleTypes) {
    Memory<uint32_t> mem32(8, 0);
    auto             mem8   = mem32.cast<uint8_t>();
    auto             mem16  = mem32.cast<uint16_t>();
    auto             mem32n = mem32.cast<uint16_t>()
                                  .cast<uint8_t>()
                                  .cast<uint16_t>()
                                  .cast<uint32_t>();

    EXPECT_EQ(mem8.size(), mem32.size() * 4);
    EXPECT_EQ(mem16.size(), mem32.size() * 2);
    EXPECT_EQ(mem32n, mem32);
}

TEST(MemoryTests, MemorySplitSingleChunk) {
    Memory<uint32_t> memory(5, 0);
    auto             splits = memory.split(1);

    EXPECT_EQ(splits.size(), 1);
    EXPECT_EQ(splits[0].size(), 5);
}

TEST(MemoryTests, MemorySplitMoreChunksThanSize) {
    Memory<uint32_t> memory(3, 0);
    EXPECT_THROW(memory.split(10), std::runtime_error);
}

TEST(MemoryTests, MemorySplitLargeChunks) {
    Memory<uint64_t> memory(100, 0);
    auto             splits = memory.split(7);

    EXPECT_EQ(splits.size(), 7);
    size_t totalSize = 0;
    for (const auto& split : splits) {
        totalSize += split.size();
    }
    EXPECT_EQ(totalSize, 100);
}

TEST(MemoryTests, MemoryElementAccess) {
    Memory<uint32_t> memory(10, 0);
    memory.iota(100);

    EXPECT_EQ(memory[0], 100);
    EXPECT_EQ(memory[5], 105);
    EXPECT_EQ(memory[9], 109);
}

TEST(MemoryTests, MemorySplitDataIntegrity) {
    Memory<uint32_t> memory(20, 0);
    memory.iota(0);
    auto splits = memory.split(4);

    for (size_t i = 0; i < splits[0].size(); ++i) {
        EXPECT_EQ(splits[0][i], i);
    }
    for (size_t i = 0; i < splits[1].size(); ++i) {
        EXPECT_EQ(splits[1][i], 5 + i);
    }
}

TEST(MemoryTests, MemoryZeroSize) {
    Memory<uint32_t> memory(0, 42);
    EXPECT_EQ(memory.size(), 0);
}

TEST(MemoryTests, MemoryDifferentTypes) {
    Memory<uint8_t>  mem8(10, 5);
    Memory<uint16_t> mem16(10, 5);
    Memory<uint64_t> mem64(10, 5);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(mem8[i], 5);
        EXPECT_EQ(mem16[i], 5);
        EXPECT_EQ(mem64[i], 5);
    }
}

TEST(ExecutorTests, Build) {
    Executor executor;

    EXPECT_FALSE(executor.hasWork());
    auto task = executor.enqueue([]() { return 42; });

    Receiver<int> rcv;
    rcv.notifier = [](const int& result) {
        std::cout << "Task completed with result: " << result << std::endl;
    };

    rcv.interrupt = [](const std::exception& e) {
        std::cout << "Task interrupted with error: " << e.what() << std::endl;
    };

    task->subscribe(rcv);

    EXPECT_TRUE(executor.hasWork());
    EXPECT_EQ(executor.snapShot(), 1);
    EXPECT_TRUE(executor.process());
    EXPECT_EQ(executor.snapShot(), 0);
    executor.kill();
    EXPECT_FALSE(executor.hasWork());
}

TEST(ExecutorTests, SingleTaskExecution) {
    Executor executor;

    EXPECT_FALSE(executor.hasWork());
    auto task = executor.enqueue([]() { return 42; });

    Receiver<int> rcv;
    rcv.notifier = [](const int& result) {
        std::cout << "Task completed with result: " << result << std::endl;
    };

    rcv.interrupt = [](const std::exception& e) {
        std::cout << "Task interrupted with error: " << e.what() << std::endl;
    };

    task->subscribe(rcv);

    EXPECT_TRUE(executor.hasWork());
    EXPECT_EQ(executor.snapShot(), 1);
    EXPECT_TRUE(executor.process());
    EXPECT_EQ(executor.snapShot(), 0);
    executor.kill();
    EXPECT_FALSE(executor.hasWork());
}

TEST(SchedulerTests, SingleTaskExecution) {
    Scheduler scheduler(2, 10);

    auto task = scheduler.enqueue([]() { return 42; });

    task->subscribe(Receiver<int>{
        .notifier =
            [](const int& result) {
                std::cout << "Task completed with result: " << result
                          << std::endl;
            },
        .interrupt =
            [](const std::exception& e) {
                std::cout << "Task interrupted with error: " << e.what()
                          << std::endl;
            }});

    EXPECT_FALSE(task->isBusy());
    EXPECT_FALSE(task->isDone());

    scheduler.launch();
    scheduler.spin();

    EXPECT_TRUE(task->isDone());
}