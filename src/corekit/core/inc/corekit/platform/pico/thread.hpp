#include <stdexcept>

namespace corekit {
    namespace system {

        struct ThreadImpl {
            template <typename Fn, typename... Args>
            requires std::invocable<Fn, Args...> ThreadImpl(Fn&& fn,
                                                            Args&&... args) {}

            void run() {
                throw std::runtime_error(
                    "Thread run not implemented on pico platform");
            }

            void join() {
                throw std::runtime_error(
                    "Thread join not implemented on pico platform");
            }
        };

    };  // namespace system
};      // namespace corekit
