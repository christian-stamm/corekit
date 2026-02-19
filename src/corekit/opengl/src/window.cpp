#include "corekit/opengl/window.hpp"

#include <cmath>
#include <format>

#include "corekit/opengl/assert.hpp"

namespace corekit {
    namespace opengl {

        using namespace corekit::types;
        using namespace corekit::utils;

        Window::Window(const Context& context)
            : Device(context.title)
            , context(context)
            , window(nullptr)
            , updateRate(0.0f) {}

        Status Window::info() const {
            corecheck(isLoaded(), "Window not loaded yet.");

            return std::format(                                           //
                "{} - {}",                                                //
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)),  //
                reinterpret_cast<const char*>(glGetString(GL_VERSION))    //
            );
        }

        void Window::resize(const Vec2& size) const {
            if (!isRunning()) {
                return;
            }

            glfwSetWindowSize(window,
                              static_cast<int>(size.x()),
                              static_cast<int>(size.y()));
        }

        void Window::update() const {
            if (!isRunning()) {
                return;
            }

            glfwSwapBuffers(window);
            glfwPollEvents();

            double dt = monitor.elapsed();
            monitor.reset();

            if (1e-9 < dt) {
                updateRate *= 0.95f;
                updateRate += 0.05f * (1.0f / dt);
            }
        }

        void Window::close() const {
            if (isRunning()) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            context.rt->kill();
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
            corecheck(context.rt->screensize.valid(), "Screensize is not set.");
            return context.rt->screensize.get();
        }

        bool Window::prepare() {
            corecheck(context.rt->screensize.valid(),
                      "No valid Screensize defined.");

            int  status = 0;
            Vec2 size   = context.rt->screensize.get();
            glfwSetErrorCallback(Window::errorHandle);

            status = glfwInit();
            corecheck(status == GLFW_TRUE, "Failed to initialize GLFW");

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
            glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);

            window =
                glfwCreateWindow(size.x(), size.y(), name.c_str(), NULL, NULL);
            corecheck(window != nullptr, "Failed to create GLFW window");

            glfwMakeContextCurrent(window);
            glfwSwapInterval(0);

            if (!context.visible) {
                glfwHideWindow(window);
            }

            glfwSetWindowUserPointer(window, this);

            glfwSetWindowCloseCallback(window, [](GLFWwindow* w) {
                void* self = glfwGetWindowUserPointer(w);
                static_cast<Window*>(self)->close();
            });

            glfwSetWindowSizeCallback(
                window,
                [](GLFWwindow* w, int width, int height) {
                    void* self = glfwGetWindowUserPointer(w);
                    auto& rt   = static_cast<Window*>(self)->context.rt;
                    // Defer resize to next frame to ensure OpenGL context is
                    // current
                    if (width > 0 && height > 0) {
                        rt->screensize.set(Vec2(width, height));
                    }
                });

            glfwSetCursorPosCallback(
                window,
                [](GLFWwindow* w, double xpos, double ypos) {
                    void* self = glfwGetWindowUserPointer(w);
                    auto& rt   = static_cast<Window*>(self)->context.rt;
                    rt->mousepos.set(Vec2(xpos, ypos));
                });

            glfwSetMouseButtonCallback(
                window,
                [](GLFWwindow* w, int button, int action, int mods) {
                    void* self = glfwGetWindowUserPointer(w);
                    auto& rt   = static_cast<Window*>(self)->context.rt;
                    Vec2  mpos = {0, 0};

                    if (rt->mousepos.valid()) {
                        mpos = rt->mousepos.get();
                    }

                    switch (button) {
                        case GLFW_MOUSE_BUTTON_LEFT:
                            mpos.x() = action == GLFW_PRESS;
                            break;
                        case GLFW_MOUSE_BUTTON_RIGHT:
                            mpos.y() = action == GLFW_PRESS;
                            break;
                        default: break;
                    }

                    rt->mousebtn.set(mpos);
                });

            status = gladLoadGLES2Loader((GLADloadproc)glfwGetProcAddress);
            corecheck(status != 0, "Failed to initialize GLAD");

            return true;
        }

        bool Window::cleanup() {
            glfwSetWindowCloseCallback(window, nullptr);
            glfwMakeContextCurrent(window);

            this->clear();
            this->close();

            glfwPollEvents();
            glfwDestroyWindow(window);
            glfwTerminate();

            this->window = nullptr;
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

    };  // namespace opengl
};      // namespace corekit
