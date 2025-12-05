#pragma once

#include "corekit/core.hpp"
#include "corekit/device/device.hpp"

#include <glad/glad.h>

namespace corekit {
namespace opengl {

    using namespace corekit::logging;
    using namespace corekit::device;

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

        Code getCode() const
        {
            return code;
        }

        GLenum getType() const
        {
            return type;
        }

        GLuint getInstance() const
        {
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

}; // namespace opengl
}; // namespace corekit