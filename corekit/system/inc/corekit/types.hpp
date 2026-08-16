#pragma once
#include <set>
#include <string>

namespace corekit {

    using uint   = unsigned int;
    using Name   = std::string;
    using Hash   = std::string;
    using Code   = std::string;
    using Status = std::string;

    namespace GPIO {
        using Pin   = uint;
        using Group = std::set<Pin>;
    };  // namespace GPIO

    namespace network {
        using Topic  = uint16_t;
        using Cookie = uint16_t;
    };  // namespace network

};  // namespace corekit