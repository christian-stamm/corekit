#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "corekit/types.hpp"
#include "corekit/utils/device.hpp"
#include "corekit/utils/math.hpp"

// KEEP GLAD BEFORE GLFW
#include <glad/glad.h>
// KEEP GLAD BEFORE GLFW
#include <GLFW/glfw3.h>
// KEEP GLAD BEFORE GLFW

namespace corekit {
    namespace render {

        using namespace corekit::types;
        using namespace corekit::utils;

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
                using List = std::vector<Settings>;

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

            static Ptr  build(const Settings& settings);
            static List build(const Settings::List& settings);

            static GLuint glRequestFBO();
            static GLuint glRequestTex();

            static void glReleaseFBO(const GLuint* fbo);
            static void glReleaseTex(const GLuint* tex);

            void verify() const;
            void bind() const;
            void unbind() const;

            virtual void resize(Vec2 size, bool force = false);

            virtual void fill(cv::Mat  image,
                              GLuint   layer = 0,
                              FillMode mode  = RESIZE_IMAGE);
            ;

            virtual void copyTo(const Ptr& target,
                                GLuint     layer  = 0,
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

    };  // namespace render
};      // namespace corekit
