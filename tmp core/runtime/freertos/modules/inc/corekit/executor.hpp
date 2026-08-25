#pragma once
#include <memory>
#include <optional>
#include <set>

#include "corekit/task.hpp"
#include "corekit/watch.hpp"

namespace corekit {

    using Timeout = std::optional<double>;

    struct Thread {
        using Ptr = std::shared_ptr<Thread>;

        Task::Ptr    task   = nullptr;
        TaskHandle_t handle = nullptr;
    };

    class Executor {
       public:
        using Ptr = std::shared_ptr<Executor>;

        Executor();
        ~Executor();

        bool enqueue(Task::Ptr task);
        void process();
        void terminate(const Timeout& timeout = std::nullopt);

       private:
        std::set<Task::Ptr> m_tasks;
        StopSource          m_stpsrc;
    };

}  // namespace corekit
