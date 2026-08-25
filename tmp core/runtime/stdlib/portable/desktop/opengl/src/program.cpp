#include "corekit/opengl/program.hpp"

#include <format>

#include "corekit/opengl/assert.hpp"

namespace corekit {
    namespace opengl {

        using namespace corekit::types;
        using namespace corekit::utils;

        Program::Program(const Settings& settings)
            : Device("Program")
            , hash(settings.hash)
            , glID(settings.glID)
            , frame(0)
            , shaders(Shader::build(settings.shaders)) {
            corecheck(!shaders.empty(),
                      "Program must have at least one shader");
        }

        Program::Ptr Program::build(const Settings& settings) {
            return std::make_shared<Program>(settings);
        }

        Program::List Program::build(const Settings::List& settings) {
            List programs;
            programs.reserve(settings.size());

            for (const Settings& settings : settings) {
                programs.push_back(build(settings));
            }

            return programs;
        }

        GLuint Program::glRequestID() {
            return glCreateProgram();
        }

        GLuint Program::glRequestVBO() {
            GLuint vbo;
            glGenBuffers(1, &vbo);
            return vbo;
        }

        GLuint Program::glRequestVAO() {
            GLuint vao;
            glGenVertexArrays(1, &vao);
            return vao;
        }

        void Program::glReleaseID(GLuint* prg) {
            glDeleteProgram(*prg);
            *prg = GL_INVALID_INDEX;
        }

        void Program::glReleaseVBO(GLuint* vbo) {
            glDeleteBuffers(1, vbo);
            *vbo = 0;
        }

        void Program::glReleaseVAO(GLuint* vao) {
            glDeleteVertexArrays(1, vao);
            *vao = 0;
        }

        bool Program::prepare() {
            bool valid = true;

            if (glID == GL_INVALID_INDEX) {
                glID = glRequestID();
            }

            if (valid) {
                valid &= compile();
            }

            if (valid) {
                valid &= link();
            }

            frame = 0;
            return valid;
        }

        bool Program::cleanup() {
            bool valid = true;

            if (valid) {
                valid &= unlink();
            }

            if (glID != GL_INVALID_INDEX) {
                glReleaseID(&glID);
                glID = GL_INVALID_INDEX;
            }

            return valid;
        }

        void Program::process() const {
            corecheck(compiled, "Program not compiled");
            corecheck(linked, "Program not linked");

            glCheckError(name);

            const GLint self = glID;
            const GLint last = selected();

            select(self);
            render();
            select(last);
            frame += 1;

            glCheckError(name);
        }

        size_t Program::getFrameNum() const {
            return size_t(frame);
        }

        GLuint Program::getInstance() const {
            return glID;
        }

        GLint Program::getUniform(const std::string& name) const {
            return glGetUniformLocation(glID, name.c_str());
        }

        GLint Program::select(GLint program) {
            glUseProgram(program);
            glCheckError();
            return selected();
        }

        GLint Program::selected() {
            GLint program;
            glGetIntegerv(GL_CURRENT_PROGRAM, &program);
            return program;
        }

        bool Program::compile() const {
            glCheckError(name);
            if (this->compiled) {
                return true;
            }

            bool success = true;
            for (const Shader::Ptr& shader : shaders) {
                success &= shader->load();
            };

            glCheckError(name);
            this->compiled = success;
            return compiled;
        }

        bool Program::link() const {
            if (this->linked) {
                return true;
            }

            glCheckError(name);

            GLint status = 0;

            for (const Shader::Ptr& shader : shaders) {
                glAttachShader(glID, shader->glID);
            }

            glLinkProgram(glID);
            glGetProgramiv(glID, GL_LINK_STATUS, &status);
            linked = (status == GL_TRUE);

            if (!linked) {
                GLint len;
                glGetProgramiv(glID, GL_INFO_LOG_LENGTH, &len);
                std::string log(len, '\0');
                glGetProgramInfoLog(glID, len, nullptr, log.data());
                logger(Level::ERROR)
                    << std::format("Program link error: {}", log);
            }

            glCheckError(name);
            return linked;
        }

        bool Program::unlink() const {
            glCheckError(name);

            for (const Shader::Ptr& shader : shaders) {
                glDetachShader(glID, shader->glID);
            };

            glCheckError(name);
            linked = false;
            return true;
        }

    };  // namespace opengl
};      // namespace corekit
