#pragma once
#include <memory>
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

            struct Settings : public Executor::Settings {
                size_t numWorkers = 1;
            };

            Scheduler(const Settings& settings, Killreq& killreq)
                : Device("Scheduler")
                , workers(settings.numWorkers, nullptr)
                , executor(Executor::build(settings, killreq))
                , killreq(killreq) {}

            template <typename Fn, typename... Args>
            Task<std::decay_t<Fn>, std::decay_t<Args>...>::Ptr enqueue(
                Fn&& fn,
                Args&&... args) const {
                return executor->enqueue(std::forward<Fn>(fn),
                                         std::forward<Args>(args)...);
            }

            size_t numWorkers() const {
                return workers.size();
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

                for (Thread::Ptr& worker : workers) {
                    worker->join();
                }

                return true;
            }

           private:
            void kill() {
                killreq.request_stop();
                executor->abort();
            }

            void daemon() {
                static uint id = 0;

                uint local_id = id++;
                logger() << "Worker " << local_id << " started.";

                while (!killreq.stop_requested()) {
                    executor->process();
                }

                logger() << "Worker " << local_id << " stopped.";
            }

            Killreq&                 killreq;
            Executor::Ptr            executor;
            std::vector<Thread::Ptr> workers;
        };

    };  // namespace system
};      // namespace corekit

namespace nlohmann {
    using namespace corekit::types;
    using namespace corekit::system;

    static void to_json(JsonMap& j, const Scheduler::Settings& cfg) {
        j = JsonMap{
            {"numWorkers", cfg.numWorkers},
            {"numTasks", cfg.maxTasks},
        };
    }

    static void from_json(const JsonMap& j, Scheduler::Settings& cfg) {
        j.at("numWorkers").get_to(cfg.numWorkers);
        j.at("numTasks").get_to(cfg.maxTasks);
    }

};  // namespace nlohmann
