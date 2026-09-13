#pragma once

#include <memory>

#include "corekit/atomic.hpp"
#include "corekit/logger.hpp"
#include "corekit/stoptoken.hpp"

namespace corekit {

    class Task {
        public:

            enum class State { READY, RUNNING, TERMINATED, ERROR };

            using Ptr = std::shared_ptr<Task>;
            Task(const std::string& name);

            Task(const Task&)            = delete;
            Task(Task&&)                 = delete;
            Task& operator=(const Task&) = delete;
            Task& operator=(Task&&)      = delete;

            virtual ~Task() = default;

            bool exec(StopToken token) noexcept;

            inline bool is_launched() const { return get_state() != State::READY; }

            inline bool is_running() const { return get_state() == State::RUNNING; }

            inline bool is_completed() const { return get_state() == State::TERMINATED; }

            inline State get_state() const { return state_.load(); }

            const std::string name;
            const Logger      logger;

        protected:

            virtual bool on_init(StopToken token) { return {}; }

            virtual bool on_exit(StopToken token) { return {}; }

            virtual bool on_exec(StopToken token) = 0;

        private:

            Atomic<State> state_;
    };

} // namespace corekit