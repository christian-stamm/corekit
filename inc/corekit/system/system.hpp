#pragma once

#include <thread_pool/thread_pool.h>

#include "corekit/system/diagnostics/logger.hpp"
#include "corekit/system/diagnostics/watch.hpp"
#include "corekit/types.hpp"

namespace corekit {

    namespace system {

        using namespace corekit::types;
        using namespace corekit::system::diagnostics;

        using Killreq    = std::stop_source;
        using ThreadPool = dp::thread_pool<>;

        Hash getEnv(const Hash& key);

        bool isRunning(const std::future<bool>& task) {
            return task.valid() && task.wait_for(std::chrono::seconds(0)) !=
                                       std::future_status::ready;
        }

        bool isDone(const std::future<bool>& task) {
            return task.valid() && task.wait_for(std::chrono::seconds(0)) ==
                                       std::future_status::ready;
        }

        struct Manager {
            struct Settings {
                Settings(size_t workers = 4, File::Path dir = RESSOURCE_DIR)
                    : numWorker(workers)
                    , workdir(dir) {}

                size_t     numWorker;
                File::Path workdir;
            };

            Manager(const Settings& settings = Settings());
            Manager& operator<<(const Settings& config);

            void shutdown();
            bool ok() const;

            ThreadPool worker;
            File::Path workdir;
            Logger     logger;
            Watch      time;

           protected:
            Killreq killreq;
        };

        inline Manager sys;

    };  // namespace system
};      // namespace corekit
