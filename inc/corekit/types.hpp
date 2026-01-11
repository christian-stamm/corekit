#pragma once

#include <filesystem>
#include <future>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <set>
#include <string>
#include <vector>

namespace corekit {

    namespace types {
        using uint = unsigned int;

        using Name   = std::string;
        using Hash   = std::string;
        using Code   = std::string;
        using Status = std::string;

        using Task    = std::future<bool>;
        using JsonMap = nlohmann::ordered_json;

        namespace File {
            using Path = std::filesystem::path;
            using List = std::vector<Path>;
        };  // namespace File

        namespace GPIO {
            using Pin   = uint;
            using Group = std::set<Pin>;
        };  // namespace GPIO

        namespace network {
            using Topic  = uint16_t;
            using Cookie = uint16_t;
        };  // namespace network

    };  // namespace types

};  // namespace corekit
