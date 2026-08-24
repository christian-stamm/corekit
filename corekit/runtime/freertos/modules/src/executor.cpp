#include "corekit/executor.hpp"

#include <corekit/time.hpp>

namespace corekit {

    Executor::Executor() {}

    Executor::~Executor() {
        terminate();
    }

    void Executor::terminate(const Timeout& timeout) {
        Watch watch(timeout);
        m_stpsrc.request_stop();

        while (!m_tasks.empty() && !watch.expired()) {
            Time::sleep(1e-3);
        }
    }

    bool Executor::enqueue(Task::Ptr task) {
        if (task == nullptr) {
            return false;
        }

        m_tasks.insert(task);
        return true;
    }

}  // namespace corekit
