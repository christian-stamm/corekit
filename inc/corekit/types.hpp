#pragma once
#include <atomic>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <set>
#include <stop_token>
#include <string>

namespace corekit {
    namespace types {

        using uint    = unsigned int;
        using Name    = std::string;
        using Hash    = std::string;
        using Code    = std::string;
        using Status  = std::string;
        using Path    = std::filesystem::path;
        using JsonMap = nlohmann::ordered_json;
        using Killreq = std::stop_source;
        using ModFlag = std::atomic<bool>;

        namespace utils {

            namespace GPIO {
                using Pin   = uint;
                using Group = std::set<Pin>;
            };  // namespace GPIO

        };  // namespace utils

        namespace network {
            using Topic  = uint16_t;
            using Cookie = uint16_t;
        };  // namespace network

    };  // namespace types
};  // namespace corekit