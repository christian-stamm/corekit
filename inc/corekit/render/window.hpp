#pragma once

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <vector>

#include "corekit/system/context.hpp"
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
        using namespace corekit::system;

        class Window : public Device {
           public:
            using Ptr  = std::shared_ptr<Window>;
            using List = std::vector<Ptr>;

            struct Settings {
                Hash title   = "<NO WINDOW TITLE>";
                Vec2 shape   = {};
                bool visible = true;
            };

            Window(const Context<Settings>& ctx);
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

           private:
            static void errorHandle(int code, const char* desc);

            virtual bool prepare() override;
            virtual bool cleanup() override;

            GLFWwindow*       window;
            Watch             monitor;
            Context<Settings> context;
            mutable float     updateRate;
        };

    };  // namespace render
};  // namespace corekit

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