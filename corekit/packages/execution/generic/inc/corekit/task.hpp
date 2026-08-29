#pragma once

#include <memory>
#include <vector>

#include "corekit/atomic.hpp"
#include "corekit/result.hpp"
#include "corekit/stoptoken.hpp"

namespace corekit {

    class Task {
       public:
        enum class State { READY, RUNNING, TERMINATED };

        using Ptr  = std::shared_ptr<Task>;
        using List = std::vector<Ptr>;

        Task();

        Task(const Task&)            = delete;
        Task(Task&&)                 = delete;
        Task& operator=(const Task&) = delete;
        Task& operator=(Task&&)      = delete;

        virtual ~Task() = default;

        VoidResult exec(StopToken token) noexcept;

        inline bool is_launched() const {
            return get_state() != State::READY;
        }

        inline bool is_running() const {
            return get_state() == State::RUNNING;
        }

        inline bool is_completed() const {
            return get_state() == State::TERMINATED;
        }

        inline State get_state() const {
            return m_state.load();
        }

       protected:
        virtual VoidResult on_enter(StopToken token) {
            return {};
        }

        virtual VoidResult on_leave(StopToken token) {
            return {};
        }

        virtual VoidResult on_run(StopToken token) = 0;

       private:
        Atomic<State> m_state;
    };

}  // namespace corekit