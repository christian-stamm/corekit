#include <cmath>
#include <format>

#include "corekit/core.hpp"

namespace corekit {
    namespace render {

        Window::Window(const Settings& settings)
            : Device(settings.title)
            , shape(settings.shape)
            , visible(settings.visible)
            , window(nullptr)
            , updateRate(0.0f) {}

        Status Window::info() const {
            corecheck(isLoaded(), "Window not loaded yet.");

            return std::format(                                           //
                "{} - {}",                                                //
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)),  //
                reinterpret_cast<const char*>(glGetString(GL_VERSION))    //
            );                                                            //
        }

        void Window::update() const {
            if (!window || !isRunning()) {
                return;
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
            this->clear();

            double dt = monitor.elapsed();
            monitor.reset();

            if (1e-9 < dt) {
                updateRate *= 0.95f;
                updateRate += 0.05f * (1.0f / dt);
            }
        }

        void Window::close() const {
            if (window)
                glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        bool Window::isRunning() const {
            if (!window)
                return false;

            return !glfwWindowShouldClose(window);
        }

        float Window::getFPS() const {
            return updateRate;
        }

        Vec2 Window::getSize() const {
            return shape;
        }

        bool Window::prepare() {
            int status = 0;
            glfwSetErrorCallback(Window::errorHandle);

            status = glfwInit();
            corecheck(status == GLFW_TRUE, "Failed to initialize GLFW");

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
            glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);

            window = glfwCreateWindow(shape.x(),
                                      shape.y(),
                                      name.c_str(),
                                      NULL,
                                      NULL);
            corecheck(window != nullptr, "Failed to create GLFW window");

            glfwMakeContextCurrent(window);
            glfwSwapInterval(0);

            if (!visible) {
                glfwHideWindow(window);
            }

            glfwSetWindowUserPointer(window, this);

            glfwSetWindowSizeCallback(
                window,
                [](GLFWwindow* w, int width, int height) {
                    void* self = glfwGetWindowUserPointer(w);
                    static_cast<Window*>(self)->shape = Vec2(width, height);
                });

            status = gladLoadGLES2Loader((GLADloadproc)glfwGetProcAddress);
            corecheck(status != 0, "Failed to initialize GLAD");

            return true;
        }

        bool Window::cleanup() {
            corecheck(window, "Window already cleaned up.");

            glfwDestroyWindow(window);
            glfwTerminate();

            window = nullptr;
            return true;
        }

        void Window::clear(const glm::vec4& color) const {
            glClearColor(color.r, color.g, color.b, color.a);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        void Window::errorHandle(int code, const char* desc) {
            std::cerr << "GLFW error callback: " << code << " - "
                      << (desc ? desc : "(null)") << std::endl;
        }

    };  // namespace render
};  // namespace corekit
