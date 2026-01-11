#pragma once
#include <cstddef>
#include <memory>
#include <tuple>

#include "corekit/system/concurrency/flow/operation.hpp"

namespace corekit {
    namespace system {
        namespace concurrency {

            struct Executor {
               public:
                using Ptr = std::shared_ptr<Executor>;

                Executor(size_t maxTasks = 128)
                    : operations(maxTasks)
                    , maxTasks(maxTasks) {}

                template <typename Fn, typename... Args>
                Task<Fn, Args...>::Ptr enqueue(Fn&& fn, Args&&... args) {
                    auto task = std::make_shared<
                        Task<std::decay_t<Fn>, std::decay_t<Args>...>>();

                    if (!operations.try_push(task)) {
                        for (const auto& subscriber : task->subs) {
                            if (subscriber.interrupt != nullptr) {
                                subscriber.interrupt(std::runtime_error(
                                    "Executor task queue is full."));
                            }
                        }

                        return nullptr;
                    }

                    task->work = std::forward<Fn>(fn);
                    task->args = std::make_tuple(std::forward<Args>(args)...);

                    return task;
                }

                bool process() {
                    Operation::Ptr task = nullptr;

                    while (operations.try_pop(task)) {
                        if (task) {
                            task->busy = true;
                            task->exec();
                            task->busy = false;
                            task->done = true;
                        }
                    }

                    return !hasWork();
                }

                void kill() {
                    operations.clear();
                }

                bool hasWork() const {
                    return 0 < snapShot();
                }

                size_t snapShot() const {
                    return operations.size();
                }

               private:
                TaskList operations;
                size_t   maxTasks;
            };

        };  // namespace concurrency
    };  // namespace system
};  // namespace corekit
