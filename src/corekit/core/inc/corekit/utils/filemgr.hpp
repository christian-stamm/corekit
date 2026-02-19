#pragma once

#include <fstream>
#include <opencv2/opencv.hpp>

#include "corekit/types.hpp"

namespace corekit {
    namespace utils {

        namespace File {

            using namespace corekit::types;
            using List = std::vector<Path>;

            std::ifstream   open(const Path& file);
            std::streamsize size(std::ifstream& stream);
            bool            exists(const Path& file);
            List            scan(const Path& dir, const std::string& ext);

            JsonMap     loadJson(const Path& path);
            std::string loadTxt(const Path& file);
            cv::Mat     loadImg(const Path& file, const bool vflip = false);

        };  // namespace File

    };  // namespace utils
};      // namespace corekit
