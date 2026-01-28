#pragma once

#include "corekit/types.hpp"
#include "corekit/utils/device.hpp"

// KEEP GLAD BEFORE GLFW
#include <glad/glad.h>
// KEEP GLAD BEFORE GLFW
#include <GLFW/glfw3.h>
// KEEP GLAD BEFORE GLFW

namespace corekit {
    namespace render {

        using namespace corekit::types;
        using namespace corekit::utils;

        class Shader : public Device {
           public:
            using Ptr  = std::shared_ptr<Shader>;
            using List = std::vector<Ptr>;

            friend class Program;

            struct Settings {
                using List = std::vector<Settings>;

                Hash   hash = "<NONE>";
                Code   code = "<NONE>";
                GLenum type = GL_INVALID_ENUM;
                GLuint glID = GL_INVALID_INDEX;
            };

            Shader(const Settings& settings);
            Shader(const Shader& other)             = delete;
            Shader(const Shader&& other)            = delete;
            Shader& operator=(const Shader& other)  = delete;
            Shader& operator=(const Shader&& other) = delete;

            static Ptr  build(const Settings& settings);
            static List build(const Settings::List& settings);

            static GLuint glRequestShd(GLenum type);
            static void   glReleaseShd(const GLuint* shd);

            Code getCode() const {
                return code;
            }

            GLenum getType() const {
                return type;
            }

            GLuint getInstance() const {
                return glID;
            }

           private:
            virtual bool prepare() override;
            virtual bool cleanup() override;

            bool compile() const;

            Hash         hash;
            Code         code;
            GLenum       type;
            GLuint       glID;
            mutable bool compiled = false;
        };

    };  // namespace render
};      // namespace corekit
