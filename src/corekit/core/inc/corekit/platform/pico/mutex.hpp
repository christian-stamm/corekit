#include <stdexcept>

namespace corekit {
    namespace system {

        struct MutexImpl {
            void lock() {
                throw std::runtime_error(
                    "Mutex lock not implemented on pico platform");
            }

            void unlock() {
                throw std::runtime_error(
                    "Mutex unlock not implemented on pico platform");
            }

            bool try_lock() {
                throw std::runtime_error(
                    "Mutex try_lock not implemented on pico platform");
            }
        };

    };  // namespace system
};      // namespace corekit
