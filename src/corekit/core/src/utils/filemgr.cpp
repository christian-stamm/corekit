#include "corekit/utils/filemgr.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace corekit {
    namespace utils {

        namespace File {

            std::ifstream open(const Path& file) {
                if (!File::exists(file)) {
                    throw std::runtime_error("File does not exist: " +
                                             file.string());
                }

                std::ifstream stream(file, std::ios::binary | std::ios::ate);
                if (!stream) {
                    throw std::runtime_error("Failed to open file: " +
                                             file.string());
                }

                if (size(stream) < 0) {
                    throw std::runtime_error("Failed to determine file size: " +
                                             file.string());
                }

                return stream;
            }

            bool exists(const Path& file) {
                return std::filesystem::exists(file);
            }

            std::streamsize size(std::ifstream& stream) {
                stream.seekg(0, std::ios::end);
                const std::streamsize fileSize = stream.tellg();
                stream.seekg(0, std::ios::beg);

                if (fileSize < 0) {
                    throw std::runtime_error("Failed to determine file size");
                }

                return fileSize;
            }

            List scan(const Path& dir, const std::string& ext) {
                List files;

                for (const auto& entry :
                     std::filesystem::directory_iterator(dir)) {
                    if (entry.is_directory()) {
                        const List subset = scan(entry.path(), ext);
                        files.insert(files.end(), subset.begin(), subset.end());
                    } else if (entry.is_regular_file()) {
                        if (entry.path().string().ends_with(ext)) {
                            files.push_back(
                                std::filesystem::absolute(entry.path()));
                        }
                    }
                }

                std::sort(files.begin(), files.end());
                return files;
            }

            JsonMap loadJson(const Path& path) {
                return nlohmann::json::parse(loadTxt(path));
            }

            std::string loadTxt(const Path& file) {
                std::ifstream stream = open(file);

                const size_t fileSize = static_cast<size_t>(size(stream));
                std::string  content(fileSize, '\0');

                if (!stream.read(content.data(), fileSize)) {
                    throw std::runtime_error("Failed to read file: " +
                                             file.string());
                }

                return content;
            }

            cv::Mat loadImg(const Path& file, const bool vflip) {
                cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);

                if (img.empty()) {
                    throw std::runtime_error("Failed to load image: " +
                                             file.string());
                }

                cv::cvtColor(img, img, cv::COLOR_RGB2RGBA);

                if (vflip) {
                    cv::flip(img, img, 0);
                }

                return img;
            }

        };  // namespace File
    };      // namespace utils
};          // namespace corekit