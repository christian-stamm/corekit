#include <thread>

namespace corekit {
    namespace system {

        struct ThreadImpl {
            std::thread thread;

            template <typename Fn, typename... Args>
            requires std::invocable<Fn, Args...> ThreadImpl(Fn&& fn,
                                                            Args&&... args)
                : thread(std::forward<Fn>(fn), std::forward<Args>(args)...) {}

            void run() {}

            void join() {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        };

    };  // namespace system
};      // namespace corekit
