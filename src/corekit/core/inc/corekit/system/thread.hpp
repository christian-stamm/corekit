#pragma once
#include <concepts>
#include <memory>
#include <vector>

#if defined(PLATFORM_PICO)
#    include "corekit/platform/pico/thread.hpp"
#elif defined(PLATFORM_UNIX)
#    include "corekit/platform/unix/thread.hpp"
#endif

namespace corekit {
    namespace system {

        template <typename T>
        concept ThreadConcept = requires(T t) {
            { t.run() } -> std::same_as<void>;
            { t.join() } -> std::same_as<void>;
        };

        static_assert(ThreadConcept<ThreadImpl>);

        class Thread {
           public:
            using Ptr  = std::shared_ptr<Thread>;
            using List = std::vector<Ptr>;

            template <typename Fn, typename... Args>
            requires std::invocable<Fn, Args...> Thread(Fn&& fn, Args&&... args)
                : impl(new ThreadImpl(std::forward<Fn>(fn),
                                      std::forward<Args>(args)...)) {}

            ~Thread();

            Thread(const Thread&)             = delete;
            Thread(const Thread&&)            = delete;
            Thread& operator=(const Thread&)  = delete;
            Thread& operator=(const Thread&&) = delete;

            void run();
            void join();

           private:
            ThreadImpl* impl;
        };

    };  // namespace system
};      // namespace corekit
