#include "corekit/core.hpp"

#include <nlohmann/json_fwd.hpp>

namespace corekit {

namespace system {

    Manager::Manager(const Settings& settings)
        : worker(settings.numWorker)
        , workdir(settings.workdir)
        , logger("System")
    {
        std::system("clear");
    }

    Manager& Manager::operator<<(const Settings& config)
    {
        std::destroy_at(&worker);
        std::construct_at(&worker, config.numWorker);

        workdir = config.workdir;
        return *this;
    }

    void Manager::shutdown()
    {
        killreq.request_stop();
    }

    bool Manager::ok() const
    {
        return !killreq.stop_requested();
    }

    void corecheck(bool condition, const std::string& message, const std::source_location& location)
    {
#ifndef NDEBUG
        if (!condition) {
            throw std::runtime_error(std::format(
                "Assertion failed:\n\n\tFile: {}\n\tFunc: {}\n\tLine: {} ({})\n\tDesc: {}", location.file_name(),
                location.function_name(), location.line(), location.column(), message));
        }
#endif
    }

    std::string getEnv(const std::string& key)
    {
        const char* value = std::getenv(key.c_str());
        corecheck(value != nullptr, "Environment variable not set: " + key);
        return value;
    }

} // namespace system

namespace math {

    template <typename T> T div(T dividend, T divisor, T fallback)
    {
        return divisor ? dividend / divisor : fallback;
    }

    int wrap(int value, int length)
    {
        if (length == 0) {
            return 0;
        }
        int r = value % length;
        if (r < 0) {
            r += length;
        }
        return r;
    }

    // Explicit instantiations (add more if needed)
    template int    div<int>(int, int, int);
    template float  div<float>(float, float, float);
    template double div<double>(double, double, double);

} // namespace math

namespace thread {

    bool isDone(const Task& result, float timeout)
    {
        static const std::chrono::nanoseconds dt(uint64_t(1e9 * timeout));
        return result.valid() && result.wait_for(dt) == std::future_status::ready;
    }

    bool isRunning(const Task& result)
    {
        return result.valid() && !isDone(result, 0.0f);
    }

} // namespace thread

namespace file {

    std::ifstream open(const Path& file)
    {
        if (!system::exists(file)) {
            throw std::runtime_error("File does not exist: " + file.string());
        }

        std::ifstream stream(file, std::ios::binary | std::ios::ate);
        if (!stream) {
            throw std::runtime_error("Failed to open file: " + file.string());
        }

        if (size(stream) < 0) {
            throw std::runtime_error("Failed to determine file size: " + file.string());
        }

        return stream;
    }

    bool exists(const Path& file)
    {
        return std::filesystem::exists(file);
    }

    std::streamsize size(std::ifstream& stream)
    {
        stream.seekg(0, std::ios::end);
        const std::streamsize fileSize = stream.tellg();
        stream.seekg(0, std::ios::beg);

        if (fileSize < 0) {
            throw std::runtime_error("Failed to determine file size");
        }

        return fileSize;
    }

    List scan(const Path& dir, const std::string& ext)
    {
        List files;

        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.is_directory()) {
                const List subset = scan(entry.path(), ext);
                files.insert(files.end(), subset.begin(), subset.end());
            }
            else if (entry.is_regular_file()) {
                if (entry.path().string().ends_with(ext)) {
                    files.push_back(std::filesystem::absolute(entry.path()));
                }
            }
        }

        std::sort(files.begin(), files.end());
        return files;
    }

    nlohmann::ordered_json loadJson(const Path& path)
    {
        return nlohmann::ordered_json::parse(loadTxt(path));
    }

    std::string loadTxt(const Path& file)
    {
        std::ifstream stream = open(file);

        const size_t fileSize = static_cast<size_t>(size(stream));
        std::string  content(fileSize, '\0');

        if (!stream.read(content.data(), fileSize)) {
            throw std::runtime_error("Failed to read file: " + file.string());
        }

        return content;
    }

    cv::Mat loadImg(const Path& file, const bool vflip)
    {
        cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);

        if (img.empty()) {
            throw std::runtime_error("Failed to load image: " + file.string());
        }

        cv::cvtColor(img, img, cv::COLOR_RGB2RGBA);

        if (vflip) {
            cv::flip(img, img, 0);
        }

        return img;
    }

} // namespace file

} // namespace corekit
