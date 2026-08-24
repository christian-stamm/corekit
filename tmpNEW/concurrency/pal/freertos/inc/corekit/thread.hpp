#pragma once
#include <memory>
#include <set>

#include "FreeRTOS.h"
#include "corekit/stoptoken.hpp"
#include "corekit/task.hpp"
#include "task.h"

inline corekit::StopSource stopSource;

inline void task_daemon(void *arg) {
    using namespace corekit;
    Task     *task  = static_cast<Task *>(arg);
    StopToken token = stopSource.get_token();

    if (task) {
        task->exec(token);
    }

    vTaskDelete(nullptr);
}

namespace corekit {

    class FreeRTOSRuntime {
       public:
        static void launch() {
            vTaskStartScheduler();
        }

        static void kill() {
            stopSource.request_stop();

            vTaskEndScheduler();
        }

        static void registerThread(Task::Ptr task) {
            tasks.insert(task);
        }

        static void unregisterThread(Task::Ptr task) {
            tasks.erase(task);
        }

       private:
        static std::set<Task::Ptr> tasks;
    };

    inline std::set<Task::Ptr> FreeRTOSRuntime::tasks;

    class FreeRTOSThread {
       public:
        using Ptr = std::shared_ptr<FreeRTOSThread>;

        FreeRTOSThread(Task::Ptr task) : task(task) {
            const BaseType_t result = xTaskCreate(task_daemon,
                                                  genTaskId().c_str(),
                                                  configMINIMAL_STACK_SIZE,
                                                  task.get(),
                                                  1,
                                                  &handle);

            if (result != pdPASS) {
                throw std::runtime_error("Failed to create FreeRTOS task");
            }

            FreeRTOSRuntime::registerThread(task);
        }

        ~FreeRTOSThread() {
            if (handle) {
                vTaskDelete(handle);
            }

            FreeRTOSRuntime::unregisterThread(task);
        }

       private:
        Task::Ptr    task;
        TaskHandle_t handle;

        static std::string genTaskId() {
            static uint64_t id = 0;
            return "Tsk-" + std::to_string(++id);
        }
    };

    using Thread = FreeRTOSThread;

}  // namespace corekit
