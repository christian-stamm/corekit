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

        std::ifstream File::open(const Path& file) {
            if (!File::exists(file)) {
                throw std::runtime_error("File does not exist: " +
                                         file.string());
            }

            std::ifstream stream(file, std::ios::binary | std::ios::ate);
            if (!stream) {
                throw std::runtime_error("Failed to open file: " +
                                         file.string());
            }

            if (File::size(stream) < 0) {
                throw std::runtime_error("Failed to determine file size: " +
                                         file.string());
            }

            return stream;
        }

        bool File::exists(const Path& file) {
            return std::filesystem::exists(file);
        }

        std::streamsize File::size(std::ifstream& stream) {
            stream.seekg(0, std::ios::end);
            const std::streamsize fileSize = stream.tellg();
            stream.seekg(0, std::ios::beg);

            if (fileSize < 0) {
                throw std::runtime_error("Failed to determine file size");
            }

            return fileSize;
        }

        File::List File::scan(const Path& dir, const std::string& ext) {
            List files;

            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (entry.is_directory()) {
                    const List subset = File::scan(entry.path(), ext);
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

        JsonMap File::loadJson(const Path& path) {
            return nlohmann::json::parse(File::loadTxt(path));
        }

        std::string File::loadTxt(const Path& file) {
            std::ifstream stream = File::open(file);

            const size_t fileSize = static_cast<size_t>(File::size(stream));
            std::string  content(fileSize, '\0');

            if (!stream.read(content.data(), fileSize)) {
                throw std::runtime_error("Failed to read file: " +
                                         file.string());
            }

            return content;
        }

        std::vector<uint8_t> File::loadBin(const Path& file) {
            std::ifstream stream = File::open(file);

            const size_t fileSize = static_cast<size_t>(File::size(stream));
            std::vector<uint8_t> content(fileSize);

            if (!stream.read(reinterpret_cast<char*>(content.data()),
                             fileSize)) {
                throw std::runtime_error("Failed to read file: " +
                                         file.string());
            }

            return content;
        }

        cv::Mat File::loadImg(const Path& file, const bool vflip) {
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

        void File::saveBin(const Path&                 file,
                           const std::vector<uint8_t>& content) {
            std::ofstream stream(file, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(content.data()),
                         content.size());
        }

        void File::saveTxt(const Path& file, const std::string& content) {
            std::ofstream stream(file);
            stream << content;
        }

        void File::saveImg(const Path& file, const cv::Mat& img) {
            cv::imwrite(file, img);
        }

        void File::saveJson(const Path& file, const JsonMap& content) {
            File::saveTxt(file, nlohmann::json(content).dump(4));
        }

    };  // namespace utils
}  // namespace corekit