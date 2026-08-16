#pragma once

#include <fstream>
#include <opencv2/opencv.hpp>

#include "corekit/types.hpp"

namespace corekit {
    namespace utils {

        using namespace corekit::types;

        struct File {
            using List = std::vector<Path>;

            File() = delete;

            static bool exists(const Path& file);
            static List scan(const Path& dir, const std::string& ext);

            static std::vector<uint8_t> loadBin(const Path& file);
            static JsonMap              loadJson(const Path& path);
            static std::string          loadTxt(const Path& file);
            static cv::Mat loadImg(const Path& file, const bool vflip = false);

            static void saveBin(const Path&                 file,
                                const std::vector<uint8_t>& content);
            static void saveJson(const Path& file, const JsonMap& content);
            static void saveTxt(const Path& file, const std::string& content);
            static void saveImg(const Path& file, const cv::Mat& img);

           protected:
            static std::ifstream   open(const Path& file);
            static std::streamsize size(std::ifstream& stream);
            static bool            existsFileDir(const Path& file);

        };  // namespace File

    };  // namespace utils
};  // namespace corekit
