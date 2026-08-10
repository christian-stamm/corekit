#include <concepts>

namespace corekit {

    template<typename T>
    concept IThread = requires(T t) {
        { t.run() } -> std::same_as<void>;
        { t.join() } -> std::same_as<void>;
        { t.joinable() } -> std::same_as<bool>;
    };

};      // namespace corekit
