#include "corekit/opengl/program.hpp"

#include "corekit/core.hpp"
#include "corekit/device/device.hpp"
#include "corekit/opengl/context.hpp"
#include "corekit/opengl/shader.hpp"

#include <format>

namespace corekit {
namespace opengl {

    using namespace corekit::system;

    Program::Program(const Settings& settings)
        : Device("Program")
        , hash(settings.hash)
        , glID(settings.glID)
        , frame(0)
        , shaders(Shader::build(settings.shaders))
    {
    }

    Program::Ptr Program::build(const Settings& settings)
    {
        return std::make_shared<Program>(settings);
    }

    Program::List Program::build(const Settings::List& settings)
    {
        List programs;

        for (const Settings& settings : settings) {
            programs.push_back(build(settings));
        }

        return programs;
    }

    bool Program::prepare()
    {
        bool valid = true;

        if (glID == GL_INVALID_INDEX) {
            glID = glRequestPrg();
        }

        if (renderLock.try_lock()) {
            for (const Program::Ptr& child : children) {
                valid &= child->prepare();
            }

            renderLock.unlock();
        }

        if (valid) {
            valid &= compile();
        }

        if (valid) {
            valid &= link();
        }

        if (valid) {
            reset();
        }

        return valid;
    }

    bool Program::cleanup()
    {
        bool valid = true;

        if (renderLock.try_lock()) {
            for (const Program::Ptr& child : children) {
                valid &= child->cleanup();
            }

            renderLock.unlock();
        }

        if (valid) {
            valid &= unlink();
        }

        if (glID != GL_INVALID_INDEX) {
            glReleasePrg(&glID);
            glID = GL_INVALID_INDEX;
        }

        reset();
        return valid;
    }

    void Program::process() const
    {
        corecheck(compiled, "Program not compiled");
        corecheck(linked, "Program not linked");

        glCheckError(name);

        const GLint self = glID;
        const GLint last = selected();

        if (renderLock.try_lock()) {

            for (const Program::Ptr& child : children) {
                child->process();
            }

            renderLock.unlock();
        }

        select(self);
        render();
        select(last);
        frame += 1;

        glCheckError(name);
    }

    void Program::reset() const
    {
        if (renderLock.try_lock()) {
            for (const Program::Ptr& child : children) {
                child->reset();
            }

            renderLock.unlock();
        }

        frame = 0;
        watch.reset();
    }

    void Program::attachDependency(const Program::Ptr& dependency)
    {
        if (dependency) {
            children.insert(dependency);
        }
    }

    void Program::detachDependency(const Program::Ptr& dependency)
    {
        if (dependency) {
            children.erase(dependency);
        }
    }

    float Program::getRuntime() const
    {
        return float(watch.elapsed());
    }

    size_t Program::getFrameNum() const
    {
        return size_t(frame);
    }

    GLuint Program::getInstance() const
    {
        return glID;
    }

    GLint Program::getUniform(const std::string& name) const
    {
        return glGetUniformLocation(glID, name.c_str());
    }

    GLint Program::select(GLint program)
    {
        glUseProgram(program);
        glCheckError();
        return selected();
    }

    GLint Program::selected()
    {
        GLint program;
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        return program;
    }

    bool Program::compile() const
    {
        glCheckError(name);
        if (this->compiled) {
            return true;
        }

        bool success = true;

        if (renderLock.try_lock()) {
            for (const Shader::Ptr& shader : shaders) {
                success &= shader->load();
            };

            renderLock.unlock();
        }

        glCheckError(name);
        this->compiled = success;
        return compiled;
    }

    bool Program::link() const
    {
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
            logger(Level::ERROR) << std::format("Program link error: {}", log);
        }

        glCheckError(name);
        return linked;
    }

    bool Program::unlink() const
    {
        glCheckError(name);

        for (const Shader::Ptr& shader : shaders) {
            glDetachShader(glID, shader->glID);
        };

        glCheckError(name);
        linked = false;
        return true;
    }

}; // namespace opengl
}; // namespace corekit
