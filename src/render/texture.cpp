#include "corekit/render/texture.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/videoio.hpp>

#include "corekit/core.hpp"
#include "corekit/utils/assert.hpp"

namespace corekit {
    namespace render {

        Texture::Texture(const Settings& settings)
            : Device("Texture")
            , hash(settings.hash)
            , size(settings.size)
            , fbo(settings.fbo)
            , tex(settings.tex)
            , type(settings.type)
            , unit(settings.unit)
            , wrap(settings.wrap)
            , intern(settings.intern)
            , filter(settings.filter)
            , flip(settings.flip)
            , srgb(settings.srgb)
            , depth(settings.depth) {}

        Texture::Ptr Texture::build(const Settings& settings) {
            return std::make_shared<Texture>(settings);
        }

        GLuint Texture::glRequestFBO() {
            GLuint fbo;
            glGenFramebuffers(1, &fbo);
            return fbo;
        }

        GLuint Texture::glRequestTex() {
            GLuint tex;
            glGenTextures(1, &tex);
            return tex;
        }

        void Texture::glReleaseFBO(GLuint* fbo) {
            glDeleteFramebuffers(1, fbo);
            *fbo = GL_INVALID_INDEX;
        }

        void Texture::glReleaseTex(GLuint* tex) {
            glDeleteTextures(1, tex);
            *tex = GL_INVALID_INDEX;
        }

        bool Texture::prepare() {
            if (fbo == GL_INVALID_INDEX) {
                this->fbo = glRequestFBO();
            }

            if (tex == GL_INVALID_INDEX) {
                this->tex = glRequestTex();
            }

            this->resize(this->size, true);
            return this->verify();
        }

        bool Texture::cleanup() {
            unbind();
            glReleaseFBO(&fbo);
            glReleaseTex(&tex);
            return true;
        }

        bool Texture::verify() const {
            if (fbo == GL_INVALID_INDEX) {
                logger.warn() << "Texture::verify => invalid FBO ID";
                return false;
            }

            if (tex == GL_INVALID_INDEX) {
                logger.warn() << "Texture::verify => invalid texture ID";
                return false;
            }

            if (size.isZero()) {
                logger.warn() << "Texture::verify => zero texture size";
                return false;
            }

            switch (type) {
                case GL_TEXTURE_2D:
                case GL_TEXTURE_3D:
                case GL_TEXTURE_2D_ARRAY:
                case GL_TEXTURE_CUBE_MAP: break;
                default: {
                    logger.warn()
                        << "Texture::verify => unsupported texture type";
                    return false;
                }
            }

            switch (intern) {
                case GL_UNSIGNED_BYTE:
                case GL_HALF_FLOAT:
                case GL_FLOAT: break;
                default: {
                    logger.warn()
                        << "Texture::verify => unsupported texture format";
                    return false;
                }
            }

            return true;
        }

        void Texture::bind() const {
            glCheckError(name);
            glActiveTexture(unit);
            glBindTexture(type, tex);

            glTexParameteri(type, GL_TEXTURE_WRAP_S, wrap);
            glTexParameteri(type, GL_TEXTURE_WRAP_T, wrap);
            if (type == GL_TEXTURE_CUBE_MAP) {
                glTexParameteri(type, GL_TEXTURE_WRAP_R, wrap);
            }
            glTexParameteri(type, GL_TEXTURE_MIN_FILTER, filter.min);
            glTexParameteri(type, GL_TEXTURE_MAG_FILTER, filter.mag);
            glCheckError(name);
        }

        void Texture::unbind() const {
            glCheckError(name);
            glActiveTexture(unit);
            glBindTexture(type, 0);
            glActiveTexture(GL_TEXTURE0);
            glCheckError(name);
        }

        void Texture::resize(Vec2 size, bool force) {
            glCheckError(name);

            if (type == GL_TEXTURE_CUBE_MAP && size.x() != size.y()) {
                logger.warn() << "Texture::resize => non-square size provided "
                                 "for cube map, discarding resize.";

                return;
            }

            if (this->size == size && !force) {
                return;
            }

            this->size = size;

            if (!verify()) {
                throw std::runtime_error(
                    "Texture::resize => texture verification "
                    "failed before resizing");
            }

            bind();

            instances.clear();

            switch (type) {
                case GL_TEXTURE_2D:
                case GL_TEXTURE_2D_ARRAY:
                case GL_TEXTURE_3D: instances = {type}; break;

                case GL_TEXTURE_CUBE_MAP:
                    instances = {GL_TEXTURE_CUBE_MAP_POSITIVE_X,
                                 GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                                 GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
                                 GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                                 GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
                                 GL_TEXTURE_CUBE_MAP_NEGATIVE_Z};
                    break;

                default:
                    throw std::runtime_error(
                        "Texture::Module => unsupported texture type");
            }

            switch (type) {
                case GL_TEXTURE_2D:
                case GL_TEXTURE_CUBE_MAP: {
                    for (GLuint instance : instances) {
                        glTexImage2D(instance,
                                     0,
                                     GL_RGBA,
                                     size.x(),
                                     size.y(),
                                     0,
                                     GL_RGBA,
                                     intern,
                                     nullptr);
                    }
                    break;
                }

                case GL_TEXTURE_3D:
                case GL_TEXTURE_2D_ARRAY: {
                    glTexImage3D(type,
                                 0,
                                 GL_RGBA,
                                 size.x(),
                                 size.y(),
                                 depth,
                                 0,
                                 GL_RGBA,
                                 intern,
                                 nullptr);
                    break;
                }

                default:
                    throw std::runtime_error(
                        "Texture::Module => unsupported texture type");
            }

            unbind();
            glCheckError(name);
        }

        void Texture::fill(cv::Mat image, GLuint layer, FillMode mode) {
            corecheck(isLoaded(),
                      "Texture needs to be loaded before filling it.");

            if (image.empty())
                throw std::runtime_error(
                    "Texture::fill => empty image provided");

            if (depth <= layer)
                throw std::runtime_error(
                    "Texture::fill => layer index out of bounds");

            glCheckError(name);
            Vec2 imsize(image.size.p[1], image.size.p[0]);

            switch (mode) {
                case RESIZE_TEXTURE: {
                    if (this->type == GL_TEXTURE_CUBE_MAP) {
                        if ((imsize.x() != imsize.y())) {
                            throw std::runtime_error(
                                "Texture::fill => non-square image size is "
                                "not supported for cube maps");
                        }
                    }

                    if (this->size != imsize) {
                        this->resize(imsize);
                    }

                    break;
                }

                case RESIZE_IMAGE: {
                    if (this->type == GL_TEXTURE_CUBE_MAP) {
                        if (this->size.x() != this->size.y()) {
                            throw std::runtime_error(
                                "Texture::fill => non-square texture size is "
                                "not supported for cube maps");
                        }
                    }

                    if (this->size != imsize) {
                        cv::resize(image,
                                   image,
                                   cv::Size(this->size.x(), this->size.y()));
                    }

                    break;
                }

                default:
                    throw std::runtime_error(
                        "Texture::fill => unknown FillMode");
            }

            bind();
            glPixelStorei(GL_UNPACK_ALIGNMENT, (image.step & 3) ? 1 : 4);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, image.step / image.elemSize());

            switch (type) {
                case GL_TEXTURE_2D:
                case GL_TEXTURE_CUBE_MAP: {
                    for (GLuint instance : instances) {
                        glTexSubImage2D(instance,
                                        0,
                                        0,
                                        0,
                                        image.cols,
                                        image.rows,
                                        GL_RGBA,
                                        intern,
                                        image.data);
                    }
                    glGenerateMipmap(type);
                    break;
                }

                case GL_TEXTURE_3D:
                case GL_TEXTURE_2D_ARRAY: {
                    glTexSubImage3D(type,
                                    0,
                                    0,
                                    0,
                                    0,
                                    image.cols,
                                    image.rows,
                                    layer,
                                    GL_RGBA,
                                    intern,
                                    image.data);
                    break;
                }

                default:
                    throw std::runtime_error(
                        "Texture::updateGPUMem => unsupported texture "
                        "type");
            }

            unbind();
            glCheckError(name);
        }

        void Texture::copyTo(const Texture& target,
                             GLuint         layer,
                             GLenum         mask,
                             GLuint         filter) const {
            glCheckError(name);

            // if (this->type != target->type) {
            //     throw std::runtime_error("Texture::copyTo => type mismatch");
            // }

            if (this->intern != target.intern) {
                throw std::runtime_error("Texture::copyTo => format mismatch");
            }

            const bool sourceLayered = (this->type == GL_TEXTURE_2D_ARRAY ||
                                        this->type == GL_TEXTURE_3D);
            const bool targetLayered = (target.type == GL_TEXTURE_2D_ARRAY ||
                                        target.type == GL_TEXTURE_3D);

            // target->resize(this->size);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, this->fbo);
            if (sourceLayered) {
                glFramebufferTextureLayer(GL_READ_FRAMEBUFFER,
                                          GL_COLOR_ATTACHMENT0,
                                          this->tex,
                                          0,
                                          layer);
            } else {
                glFramebufferTexture2D(GL_READ_FRAMEBUFFER,
                                       GL_COLOR_ATTACHMENT0,
                                       instances.front(),
                                       this->tex,
                                       0);
            }

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target.fbo);
            if (targetLayered) {
                glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER,
                                          GL_COLOR_ATTACHMENT0,
                                          target.tex,
                                          0,
                                          layer);
            } else {
                glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
                                       GL_COLOR_ATTACHMENT0,
                                       target.instances.front(),
                                       target.tex,
                                       0);
            }

            // Copy at render resolution (no scaling here)
            glBlitFramebuffer(0,
                              0,
                              this->size.x(),
                              this->size.y(),
                              0,
                              0,
                              target.size.x(),
                              target.size.y(),
                              mask,
                              filter);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glGenerateMipmap(target.type);
            target.unbind();

            glCheckError(name);
        }

        void Texture::copyTo(const Ptr& target,
                             GLuint     layer,
                             GLenum     mask,
                             GLuint     filter) const {
            glCheckError(name);

            copyTo(*target.get(), layer, mask, filter);
        }

        uint Texture::getSlot() const {
            return unit - GL_TEXTURE0;
        }

    };  // namespace render
};  // namespace corekit
