#pragma once
#include <memory>
#include <tuple>

#include "corekit/types.hpp"
#include "corekit/utils/task.hpp"

namespace corekit {
    namespace system {

        using namespace corekit::types;
        using namespace corekit::utils;

        struct Executor {
           public:
            using Ptr = std::shared_ptr<Executor>;

            struct Settings {
                size_t maxTasks = 1;
            };

            Executor(const Settings& settings, Killreq& killreq)
                : operations(settings.maxTasks)
                , killreq(killreq) {}

            static Ptr build(const Settings& settings, Killreq& killreq) {
                return std::make_shared<Executor>(settings, killreq);
            }

            template <typename Fn, typename... Args>
            Task<std::decay_t<Fn>, std::decay_t<Args>...>::Ptr enqueue(
                Fn&& fn,
                Args&&... args) {
                using TaskType = Task<std::decay_t<Fn>, std::decay_t<Args>...>;
                auto task      = std::make_shared<TaskType>();

                // Set work and args BEFORE pushing to queue
                task->work = std::forward<Fn>(fn);
                task->args = std::make_tuple(std::forward<Args>(args)...);

                if (!operations.try_push(task)) {
                    for (const auto& subscriber : task->subs) {
                        if (subscriber.interrupt != nullptr) {
                            subscriber.interrupt(std::runtime_error(
                                "Executor task queue is full."));
                        }
                    }

                    return nullptr;
                }

                return task;
            }

            bool process() {
                Operation::Ptr task = nullptr;

                while (operations.try_pop(task) && !killreq.stop_requested()) {
                    if (task) {
                        task->busy = true;
                        task->exec();
                        task->busy = false;
                        task->done = true;
                    }
                }

                return !busy();
            }

            void abort() {
                operations.clear();
            }

            bool busy() const {
                return 0 < tasks();
            }

            size_t tasks() const {
                return operations.size();
            }

           private:
            TaskList operations;
            Killreq& killreq;
        };

    };  // namespace system
};      // namespace corekit
