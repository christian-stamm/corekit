#pragma once

#include <filesystem>
#include <future>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <set>
#include <stop_token>
#include <string>
#include <vector>

namespace corekit {

    namespace utils {
        namespace File {
            using Path = std::filesystem::path;
            using List = std::vector<Path>;
        };  // namespace File
    };      // namespace utils

    namespace device {
        namespace GPIO {
            using Pin   = uint;
            using Group = std::set<Pin>;
        };  // namespace GPIO
    };      // namespace device

    namespace network {
        using Topic  = uint16_t;
        using Cookie = uint16_t;
    };  // namespace network

    namespace types {
        using namespace corekit::utils;
        using namespace corekit::device;

        using uint = unsigned int;

        using Name   = std::string;
        using Hash   = std::string;
        using Code   = std::string;
        using Status = std::string;

        using JsonMap = nlohmann::ordered_json;
        using Killreq = std::stop_source;

    };  // namespace types

};  // namespace corekit
