#pragma once

#include <concepts>

#include "corekit/platform/stoptoken.hpp"

namespace corekit {

    using StopToken  = platform::StopToken;
    using StopSource = platform::StopSource;

    template <typename T>
    concept StopTokenLike =  //
        requires(const T& token) {
            { token.stop_requested() } -> std::convertible_to<bool>;
            { token.stop_possible() } -> std::convertible_to<bool>;
        };

    template <typename T>
    concept StopSourceLike =  //
        requires(T& source, const T& csource) {
            { source.request_stop() } -> std::convertible_to<bool>;
            { csource.stop_requested() } -> std::convertible_to<bool>;
            { csource.stop_possible() } -> std::convertible_to<bool>;
            { csource.get_token() } -> std::convertible_to<StopToken>;
        };

}  // namespace corekit