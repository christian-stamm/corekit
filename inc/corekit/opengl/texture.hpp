#pragma once

#include <glad/glad.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "corekit/core.hpp"
#include "corekit/device/device.hpp"
#include "corekit/structs/vector.hpp"

namespace corekit {
    namespace opengl {

        using namespace corekit::device;
        using namespace corekit::structs;

        class Texture : public Device {
           public:
            using Ptr  = std::shared_ptr<Texture>;
            using List = std::vector<Ptr>;

            enum FillMode {
                RESIZE_TEXTURE,
                RESIZE_IMAGE,
            };

            struct Filter {
                GLuint min = GL_LINEAR;
                GLuint mag = GL_LINEAR;
            };

            struct Settings {
                Hash   hash   = "<NONE>";          //
                Vec2   size   = Vec2();            //
                GLuint fbo    = GL_INVALID_INDEX;  //
                GLuint tex    = GL_INVALID_INDEX;  //
                GLuint type   = GL_TEXTURE_2D;     //
                GLuint unit   = GL_TEXTURE0;       //
                GLuint wrap   = GL_CLAMP_TO_EDGE;  //
                Filter filter = {};                //
                GLuint intern = GL_UNSIGNED_BYTE;  //
                bool   flip   = false;             //
                bool   srgb   = false;             //
                uint   depth  = 1;                 //
            };

            Texture(const Settings& settings);
            Texture(const Texture& other)             = delete;
            Texture(const Texture&& other)            = delete;
            Texture& operator=(const Texture& other)  = delete;
            Texture& operator=(const Texture&& other) = delete;

            void reconfigure();
            void verify() const;
            void bind() const;
            void unbind() const;
            void fill(cv::Mat  image,
                      uint     layer = 0,
                      FillMode mode  = RESIZE_IMAGE);
            void copyTo(const Ptr& target,
                        GLenum     mask   = GL_COLOR_BUFFER_BIT,
                        GLuint     filter = GL_NEAREST) const;

            Hash   hash;
            Vec2   size;
            GLuint fbo;
            GLuint tex;
            GLuint type;
            GLuint intern;
            GLuint unit;
            GLuint wrap;
            Filter filter;
            bool   flip;
            bool   srgb;
            uint   depth;

           protected:
            virtual bool prepare() override;
            virtual bool cleanup() override;

            std::vector<GLuint> instances;
        };

    };  // namespace opengl
};  // namespace corekit
