#pragma once

#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "corekit/device/device.hpp"
#include "corekit/logging/watch.hpp"
#include "corekit/opengl/shader.hpp"
#include "corekit/structs/mutex.hpp"

namespace corekit {
    namespace opengl {

        using namespace corekit::device;
        using namespace corekit::structs;

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

            void process() const;
            void reset() const;

            virtual void attachDependency(const Program::Ptr& dep);
            virtual void detachDependency(const Program::Ptr& dep);

            float  getRuntime() const;
            size_t getFrameNum() const;
            GLuint getInstance() const;

            Hash         hash;
            Shader::List shaders;
            Program::Set children;

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

            mutable Watch  watch;
            mutable uint   frame;
            mutable IMutex renderLock;

            GLuint       glID     = GL_INVALID_INDEX;
            mutable bool compiled = false;
            mutable bool linked   = false;
        };

    };  // namespace opengl
};  // namespace corekit
