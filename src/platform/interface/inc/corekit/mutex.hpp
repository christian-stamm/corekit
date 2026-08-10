#include <concepts>

namespace corekit {

    template<typename T>
    concept IMutex = requires(T t) {
        { t.lock() } -> std::same_as<void>;
        { t.unlock() } -> std::same_as<void>;
        { t.try_lock() } -> std::same_as<bool>;
    };

};  // namespace corekit

