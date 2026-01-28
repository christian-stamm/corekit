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

            struct Settings {
                size_t numWorkers = 4;
                size_t numTasks   = 64;
            };

            Scheduler(const Settings& settings, Killreq& killreq)
                : Device("Scheduler")
                , workers(settings.numWorkers, nullptr)
                , executor(Executor::build(killreq, settings.numTasks))
                , killreq(killreq) {}

            template <typename Fn, typename... Args>
            Task<std::decay_t<Fn>, std::decay_t<Args>...>::Ptr enqueue(
                Fn&& fn,
                Args&&... args) {
                return executor->enqueue(std::forward<Fn>(fn),
                                         std::forward<Args>(args)...);
            }

            size_t numWorkers() const {
                return workers.size();
            }

            void kill() {
                killreq.request_stop();
                executor->abort();

                for (const Thread::Ptr& worker : workers) {
                    worker->join();
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

            Killreq&                 killreq;
            Executor::Ptr            executor;
            std::vector<Thread::Ptr> workers;
        };

    };  // namespace system
};  // namespace corekit

namespace nlohmann {
    using namespace corekit::types;
    using namespace corekit::system;

    static void to_json(JsonMap& j, const Scheduler::Settings& cfg) {
        j = JsonMap{
            {"numWorkers", cfg.numWorkers},
            {"numTasks", cfg.numTasks},
        };
    }

    static void from_json(const JsonMap& j, Scheduler::Settings& cfg) {
        j.at("numWorkers").get_to(cfg.numWorkers);
        j.at("numTasks").get_to(cfg.numTasks);
    }

};  // namespace nlohmann
