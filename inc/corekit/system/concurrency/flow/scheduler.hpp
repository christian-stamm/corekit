#pragma once
#include <cstddef>
#include <memory>
#include <stop_token>
#include <vector>

#include "corekit/system/concurrency/flow/executor.hpp"
#include "corekit/system/concurrency/thread.hpp"

namespace corekit {
    namespace system {
        namespace concurrency {

            struct Scheduler {
               public:
                using Ptr     = std::shared_ptr<Scheduler>;
                using Killreq = std::stop_source;

                Scheduler(size_t numWorkers = 1, size_t maxTasks = 128)
                    : executor(std::make_shared<Executor>(maxTasks))
                    , workers(numWorkers, nullptr) {
                    for (Thread::Ptr& worker : workers) {
                        worker = std::make_shared<Thread>(
                            [this]() { this->daemon(); });
                    }
                }

                ~Scheduler() {
                    kill();
                }

                template <typename Fn, typename... Args>
                Task<std::decay_t<Fn>, std::decay_t<Args>...>::Ptr enqueue(
                    Fn&& fn,
                    Args&&... args) {
                    return executor->enqueue(std::forward<Fn>(fn),
                                             std::forward<Args>(args)...);
                }

                void launch() {
                    for (const Thread::Ptr& worker : workers) {
                        worker->run();
                    }
                }

                void spin() {
                    for (const Thread::Ptr& worker : workers) {
                        worker->join();
                    }
                }

                void kill() {
                    killreq.request_stop();
                    executor->kill();
                    this->spin();
                    workers.clear();
                }

                size_t size() const {
                    return workers.size();
                }

               private:
                void daemon() {
                    while (!killreq.stop_requested()) {
                        executor->process();
                    }
                }

                Killreq                  killreq;
                Executor::Ptr            executor;
                std::vector<Thread::Ptr> workers;
            };

        };  // namespace concurrency
    };      // namespace system
};          // namespace corekit
