#pragma once

#include <thread_pool/thread_pool.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <source_location>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

#include "corekit/logging/logger.hpp"

#ifndef RESSOURCE_DIR
#    error "RESSOURCE_DIR not defined"
#endif

namespace corekit {

    namespace types {

        using Name     = std::string;
        using Hash     = std::string;
        using Path     = std::filesystem::path;
        using FileList = std::vector<Path>;

    };  // namespace types

    namespace thread {

        using Task = std::future<bool>;

        bool isDone(const Task& task, float timeout = 0.0f);
        bool isRunning(const Task& task);

    };  // namespace thread

    namespace file {

        using Path = std::filesystem::path;
        using List = std::vector<Path>;

        std::ifstream   open(const Path& file);
        std::streamsize size(std::ifstream& stream);
        bool            exists(const Path& file);
        List            scan(const Path& dir, const std::string& ext);

        nlohmann::ordered_json loadJson(const Path& path);
        std::string            loadTxt(const Path& file);
        cv::Mat loadImg(const Path& file, const bool vflip = false);

    };  // namespace file

    namespace opengl {

        using Hash   = std::string;
        using Status = std::string;
        using Code   = std::string;

    };  // namespace opengl

    namespace network {

        using Topic  = uint16_t;
        using Cookie = uint16_t;

    };  // namespace network

    namespace math {

        template <typename T>
        T   div(T dividend, T divisor, T fallback);
        int wrap(int value, int length);

    };  // namespace math

    namespace system {

        using namespace corekit::file;
        using namespace corekit::logging;

        using Killreq    = std::stop_source;
        using ThreadPool = dp::thread_pool<>;

        std::string getEnv(const std::string& key);

        void corecheck(bool               condition,
                       const std::string& message = "<NO DESCRIPTION>",  //
                       const std::source_location& location =
                           std::source_location::current()  //
        );

        struct Manager {
            struct Settings {
                Settings(size_t workers = 4, Path dir = RESSOURCE_DIR)
                    : numWorker(workers)
                    , workdir(dir) {}

                size_t numWorker;
                Path   workdir;
            };

            Manager(const Settings& settings = Settings());
            Manager& operator<<(const Settings& config);

            void shutdown();
            bool ok() const;

            ThreadPool worker;
            Path       workdir;
            Logger     logger;

           protected:
            Killreq killreq;
        };

        inline Manager sys;

    };  // namespace system
};  // namespace corekit
