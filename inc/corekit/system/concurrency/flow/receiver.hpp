#pragma once
#include <exception>
#include <functional>

namespace corekit {
    namespace system {
        namespace concurrency {

            template <typename T>
            struct Receiver {
                using KillEvent   = std::function<void(const std::exception&)>;
                using NotifyEvent = std::function<void(const T& result)>;

                NotifyEvent notifier  = nullptr;
                KillEvent   interrupt = nullptr;
            };

        };  // namespace concurrency
    };  // namespace system
};  // namespace corekit
