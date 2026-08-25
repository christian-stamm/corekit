#pragma once

#include <expected>
#include <memory>

#include "corekit/atomic.hpp"
#include "corekit/stoptoken.hpp"

namespace corekit::runtime {

    struct Error {
        std::string message;
        uint32_t    code;

        bool operator==(const Error& other) const {
            return message == other.message && code == other.code;
        }
    };

    using Result = std::expected<void, Error>;

    template <typename Task, typename StopToken>
    concept TaskLike =
        StoptokenLike<StopToken> && requires(Task task, StopToken token) {
            { task.exec(token) } -> std::convertible_to<Result>;
            { task.is_launched() } -> std::convertible_to<bool>;
            { task.is_running() } -> std::convertible_to<bool>;
        };

    class Task {
       public:
        enum class State { READY, RUNNING, COMPLETED, FAILED, CRASHED };

        using Result = std::expected<void, std::exception_ptr>;
        using Ptr    = std::shared_ptr<Task>;

        Task();

        Task(const Task&)            = delete;
        Task(Task&&)                 = delete;
        Task& operator=(const Task&) = delete;
        Task& operator=(Task&&)      = delete;

        virtual ~Task() = default;

        Result exec(StopToken token);

        inline bool is_launched() const {
            return get_state() != State::READY;
        }

        inline bool is_crashed() const {
            return get_state() == State::CRASHED;
        }

        inline bool is_failed() const {
            return get_state() == State::FAILED;
        }

        inline bool is_running() const {
            return get_state() == State::RUNNING;
        }

        inline bool is_completed() const {
            return get_state() == State::COMPLETED;
        }

        inline State get_state() const {
            return m_state.load();
        }

       protected:
        virtual Result on_enter(StopToken token) {
            return {};
        }

        virtual Result on_leave(StopToken token) {
            return {};
        }

        virtual Result on_run(StopToken token) = 0;

        virtual void on_exception(std::exception_ptr eptr) noexcept {}

       private:
        Atomic<State> m_state;
    };

}  // namespace corekit::runtime