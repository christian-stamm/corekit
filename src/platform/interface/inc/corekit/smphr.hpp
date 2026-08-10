#include <concepts>

namespace corekit {
   
    template<typename T>
    concept ISemaphore = requires(T t) {
        { t.acquire() } -> std::same_as<void>;
        { t.release() } -> std::same_as<void>;
        { t.try_acquire() } -> std::same_as<bool>;
    };

};      // namespace corekit
