#pragma once
#include "corekit/system/conf/observer.hpp"
#include "corekit/system/flow/scheduler.hpp"
#include "corekit/types.hpp"
#include "corekit/utils/math.hpp"

namespace corekit {

    namespace system {

        using namespace corekit::types;
        using namespace corekit::utils;

        struct Runtime {
            using Ptr = std::shared_ptr<Runtime>;

            Runtime(const Scheduler::Settings& scheduler = {})
                : scheduler(std::make_shared<Scheduler>(scheduler, killreq)) {}

            static Ptr build(const Scheduler::Settings& scheduler = {}) {
                return std::make_shared<Runtime>(scheduler);
            }

            bool ok() const {
                return !killreq.stop_requested() && scheduler->isLoaded();
            }

            void launch() {
                scheduler->load();
            }

            void kill() const {
                scheduler->unload();
            }

            const Scheduler::Ptr scheduler;
            Observable<Vec2>     mousepos;
            Observable<Vec2>     mousebtn;
            Observable<Vec2>     screensize;

           protected:
            Killreq killreq;
        };

    };  // namespace system
};      // namespace corekit
