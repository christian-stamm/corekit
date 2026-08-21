#pragma once

#include "corekit/platform/token.hpp"

namespace corekit {

    using StopToken  = platform::StopToken;
    using StopSource = platform::StopSource;

    template <typename T>
    concept StopTokenLike =  //
        requires(T token) {
            { token.stop_requested() } -> std::convertible_to<bool>;
            { token.stop_possible() } -> std::convertible_to<bool>;
        };

    template <typename T>
    concept StopSourceLike =  //
        requires(T source) {
            { source.stop_requested() } -> std::convertible_to<bool>;
            { source.stop_possible() } -> std::convertible_to<bool>;
            { source.request_stop() } -> std::convertible_to<bool>;
            { source.get_token() } -> std::convertible_to<StopToken>;
        };

    static_assert(  //
        StopTokenLike<StopToken>,
        "Implementation of StopToken does not satisfy StopTokenLike");

    static_assert(  //
        StopSourceLike<StopSource, StopToken>,
        "Implementation of StopSource does not satisfy StopSourceLike");

}  // namespace corekit