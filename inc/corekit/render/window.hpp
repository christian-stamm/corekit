#pragma once

#include <glm/glm.hpp>
#include <nlohmann/json_fwd.hpp>
#include <vector>

#include "corekit/types.hpp"
#include "corekit/utils/device.hpp"
#include "corekit/utils/math.hpp"
#include "corekit/utils/watch.hpp"

// KEEP GLAD BEFORE GLFW
#include <glad/glad.h>
// KEEP GLAD BEFORE GLFW
#include <GLFW/glfw3.h>
// KEEP GLAD BEFORE GLFW

namespace corekit {
    namespace render {

        using namespace corekit::types;
        using namespace corekit::utils;

        class Window : public Device {
            using CloseHandle      = std::function<void()>;
            using ResizeHandle     = std::function<void(Vec2)>;
            using MouseMoveHandle  = std::function<void(Vec2)>;
            using MouseClickHandle = std::function<void(bool, bool)>;
            using KeyPressHandle   = std::function<void(int, bool)>;

           public:
            using Ptr  = std::shared_ptr<Window>;
            using List = std::vector<Ptr>;

            struct Notifier {
                using List = std::vector<Notifier>;

                CloseHandle      onClose      = nullptr;
                ResizeHandle     onResize     = nullptr;
                MouseMoveHandle  onMouseMove  = nullptr;
                MouseClickHandle onMouseClick = nullptr;
                KeyPressHandle   onKeyPress   = nullptr;
            };

            struct Settings {
                Hash title   = "<NO TITLE>";
                Vec2 shape   = Vec2(800, 600);
                bool visible = true;
            };

            Window(const Settings& settings);
            Window(const Window& other)             = delete;
            Window(const Window&& other)            = delete;
            Window& operator=(const Window& other)  = delete;
            Window& operator=(const Window&& other) = delete;

            Status info() const;
            void   close() const;
            void clear(const glm::vec4& color = {0.0f, 0.0f, 0.0f, 1.0f}) const;
            void update() const;

            float getFPS() const;
            Vec2  getSize() const;

            bool isRunning() const;
            bool isVisible() const;
            void addNotifier(const Notifier& notifier);

           private:
            static void errorHandle(int code, const char* desc);

            virtual bool prepare() override;
            virtual bool cleanup() override;

            Vec2           shape;
            bool           visible;
            GLFWwindow*    window;
            Watch          monitor;
            mutable float  updateRate;
            Notifier::List notifiers;
        };

    };  // namespace render
};      // namespace corekit

namespace nlohmann {
    using namespace corekit::types;
    using namespace corekit::utils;
    using namespace corekit::render;

    inline void from_json(const JsonMap& j, Window::Settings& s) {
        j.at("title").get_to(s.title);
        j.at("shape").get_to(s.shape);
        j.at("visible").get_to(s.visible);
    }

    inline void to_json(JsonMap& j, const Window::Settings& s) {
        j = JsonMap{
            {"title", s.title},
            {"shape", s.shape},
            {"visible", s.visible},
        };
    }

};  // namespace nlohmann