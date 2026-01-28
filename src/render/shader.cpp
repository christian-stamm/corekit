#include <format>

#include "corekit/core.hpp"

namespace corekit {
    namespace render {

        Shader::Shader(const Settings& settings)
            : Device("Shader")
            , hash(settings.hash)
            , code(settings.code)
            , type(settings.type)
            , glID(settings.glID) {}

        Shader::Ptr Shader::build(const Shader::Settings& settings) {
            return std::make_shared<Shader>(settings);
        }

        Shader::List Shader::build(const Shader::Settings::List& settings) {
            Shader::List shaders;
            shaders.reserve(settings.size());

            for (const Settings& settings : settings) {
                shaders.push_back(build(settings));
            }

            return shaders;
        }

        GLuint Shader::glRequestShd(GLenum type) {
            return glCreateShader(type);
        }

        void Shader::glReleaseShd(const GLuint* shd) {
            glDeleteShader(*shd);
        }

        bool Shader::prepare() {
            if (glID == GL_INVALID_INDEX) {
                this->glID = glRequestShd(type);
            }

            return compile();
        }

        bool Shader::cleanup() {
            if (glID != GL_INVALID_INDEX) {
                glReleaseShd(&glID);
                glID = GL_INVALID_INDEX;
            }

            return true;
        }

        bool Shader::compile() const {
            glCheckError(name);
            if (this->compiled) {
                return true;
            }

            GLint status = 0;

            const GLchar* code = this->code.c_str();
            glShaderSource(glID, 1, &code, nullptr);
            glCompileShader(glID);
            glGetShaderiv(glID, GL_COMPILE_STATUS, &status);
            compiled = (status == GL_TRUE);

            if (!compiled) {
                GLint len;
                glGetShaderiv(glID, GL_INFO_LOG_LENGTH, &len);
                std::string log(len, '\0');
                glGetShaderInfoLog(glID, len - 1, nullptr, log.data());
                logger(Level::ERROR)
                    << std::format("Cannot compile '{}': {}", name, log);
            }

            glCheckError(name);
            return compiled;
        }

    };  // namespace render
};      // namespace corekit
