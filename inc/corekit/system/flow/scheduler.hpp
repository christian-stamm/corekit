#pragma once
#include <cstddef>
#include <memory>
#include <stop_token>
#include <vector>

#include "corekit/system/flow/executor.hpp"
#include "corekit/system/sync/thread.hpp"
#include "corekit/types.hpp"
#include "corekit/utils/device.hpp"

namespace corekit {
    namespace system {

        using namespace corekit::types;
        using namespace corekit::utils;

        struct Scheduler : public Device {
           public:
            using Ptr = std::shared_ptr<Scheduler>;

            Scheduler(Name name, size_t numWorkers = 1, size_t maxTasks = 128)
                : Device(name)
                , executor(std::make_shared<Executor>(maxTasks))
                , workers(numWorkers, nullptr) {}

            template <typename Fn, typename... Args>
            Task<std::decay_t<Fn>, std::decay_t<Args>...>::Ptr enqueue(
                Fn&& fn,
                Args&&... args) {
                return executor->enqueue(std::forward<Fn>(fn),
                                         std::forward<Args>(args)...);
            }

            size_t numWorker() const {
                return workers.size();
            }

            bool ok() const {
                return !killreq.stop_requested();
            }

            void kill() {
                if (!killreq.stop_requested()) {
                    killreq.request_stop();
                    executor->kill();

                    for (const Thread::Ptr& worker : workers) {
                        worker->join();
                    }
                }
            }

           protected:
            virtual bool prepare() {
                for (Thread::Ptr& worker : workers) {
                    worker =
                        std::make_shared<Thread>([this]() { this->daemon(); });
                    worker->run();
                }

                return true;
            }

            virtual bool cleanup() {
                this->kill();
                workers.clear();
                return true;
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

    };  // namespace system
};      // namespace corekit
