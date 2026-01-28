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
                "DEV: {} - {}",                                           //
                reinterpret_cast<const char*>(glGetString(GL_RENDERER)),  //
                reinterpret_cast<const char*>(glGetString(GL_VERSION))    //
            );                                                            //
        }

        void Window::update() const {
            if (!window) {
                return;
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
            clear(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

            double dt = monitor.elapsed();
            monitor.reset();

            updateRate *= 0.9f;
            updateRate += 0.1f * (1.0f / dt);
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

            glfwSetWindowCloseCallback(window, [](GLFWwindow* w) {
                Window* window =
                    static_cast<Window*>(glfwGetWindowUserPointer(w));

                if (window) {
                    for (const Notifier& notifier : window->notifiers) {
                        if (notifier.onClose) {
                            notifier.onClose();
                        }
                    }
                }
            });

            glfwSetWindowSizeCallback(
                window,
                [](GLFWwindow* w, int width, int height) {
                    Window* window =
                        static_cast<Window*>(glfwGetWindowUserPointer(w));

                    if (window) {
                        window->shape = Vec2(width, height);

                        for (const Notifier& notifier : window->notifiers) {
                            if (notifier.onResize) {
                                notifier.onResize(window->shape);
                            }
                        }
                    }
                });

            status = gladLoadGLES2Loader((GLADloadproc)glfwGetProcAddress);
            corecheck(status != 0, "Failed to initialize GLAD");

            return true;
        }

        bool Window::cleanup() {
            corecheck(window, "Window already cleaned up.");

            close();
            glfwDestroyWindow(window);
            glfwTerminate();

            return window == nullptr;
        }

        void Window::clear(const glm::vec4& color) const {
            glClearColor(color.r, color.g, color.b, color.a);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        void Window::addNotifier(const Notifier& notifier) {
            notifiers.push_back(notifier);
        }

        void Window::errorHandle(int code, const char* desc) {
            std::cerr << "GLFW error callback: " << code << " - "
                      << (desc ? desc : "(null)") << std::endl;
        }

    };  // namespace render
};      // namespace corekit
