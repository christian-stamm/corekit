#pragma once

#include <set>
#include <string>
#include <vector>

#include "corekit/render/shader.hpp"
#include "corekit/system/sync/mutex.hpp"
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
        using namespace corekit::system;

        class Program : public Device {
           public:
            using Ptr  = std::shared_ptr<Program>;
            using Set  = std::set<Ptr>;
            using List = std::vector<Ptr>;

            struct Settings {
                using List       = std::vector<Settings>;
                using ShaderList = Shader::Settings::List;

                Hash       hash    = "<NONE>";
                ShaderList shaders = {};
                GLuint     glID    = GL_INVALID_INDEX;
            };

            Program(const Settings& settings);
            Program(const Program& other)             = delete;
            Program(const Program&& other)            = delete;
            Program& operator=(const Program& other)  = delete;
            Program& operator=(const Program&& other) = delete;

            static Ptr  build(const Settings& settings);
            static List build(const Settings::List& settings);

            static GLuint glRequestID();
            static GLuint glRequestVBO();
            static GLuint glRequestVAO();

            static void glReleaseID(GLuint* prg);
            static void glReleaseVBO(GLuint* vbo);
            static void glReleaseVAO(GLuint* vao);

            virtual void process() const;

            size_t getFrameNum() const;
            GLuint getInstance() const;

            Hash         hash;
            Shader::List shaders;

           protected:
            virtual bool prepare() override;
            virtual bool cleanup() override;
            virtual void render() const {};

            GLint getUniform(const std::string& name) const;

           private:
            static GLint select(GLint program);
            static GLint selected();

            bool compile() const;
            bool link() const;
            bool unlink() const;

            mutable uint frame;

            GLuint       glID     = GL_INVALID_INDEX;
            mutable bool compiled = false;
            mutable bool linked   = false;
        };

    };  // namespace render
};  // namespace corekit
