
#include <iostream>

#include "corekit/thread.hpp"
#include "corekit/time.hpp"

extern "C" void vApplicationMallocFailedHook(void) {
    // Fail loudly during tests.
    __builtin_trap();
}

extern "C" void vApplicationStackOverflowHook(TaskHandle_t xTask,
                                              char*        pcTaskName) {
    // Fail loudly during tests.
    __builtin_trap();
}

extern "C" void vApplicationDaemonTaskStartupHook(void) {
    // Nothing required for tests.
}

namespace corekit {

    class TestTask : public Task {
       public:
        TestTask(std::string name) : name(name) {
            std::cout << "TestTask " << name << " created." << std::endl;
        }

        ~TestTask() {
            std::cout << "TestTask " << name << " destroyed." << std::endl;
        }

       protected:
        virtual bool on_run(const StopToken& token) override {
            while (!token.stop_requested()) {
                std::cout << "TestTask " << name << " is running..."
                          << std::endl;

                Time::sleep(1e-6);
            }

            return true;
        }

        virtual bool on_enter(const StopToken& token) override {
            std::cout << "TestTask " << name << " is entering..." << std::endl;
            return true;
        }

        virtual bool on_leave(const StopToken& token) override {
            std::cout << "TestTask " << name << " is leaving..." << std::endl;
            return true;
        }

        std::string name;
    };

}  // namespace corekit

int main(void) {
    using namespace corekit;

    TestTask::Ptr task1 = std::make_shared<TestTask>("Task 1");
    TestTask::Ptr task2 = std::make_shared<TestTask>("Task 2");

    Thread::Ptr thread1 = std::make_shared<FreeRTOSThread>(task1);
    Thread::Ptr thread2 = std::make_shared<FreeRTOSThread>(task2);

    FreeRTOSRuntime::launch();

    return 0;
}