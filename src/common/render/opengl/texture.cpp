#include "corekit/render/opengl/texture.hpp"

#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "corekit/system/diagnostics/assert.hpp"

namespace corekit {
    namespace render {
        namespace opengl {

            using namespace corekit::system::diagnostics;

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

            Texture::List Texture::build(const Settings::List& settings) {
                Texture::List textures;
                textures.reserve(settings.size());

                for (const auto& setting : settings) {
                    textures.push_back(build(setting));
                }

                return textures;
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

            void Texture::glReleaseFBO(const GLuint* fbo) {
                glDeleteFramebuffers(1, fbo);
            }

            void Texture::glReleaseTex(const GLuint* tex) {
                glDeleteTextures(1, tex);
            }

            bool Texture::prepare() {
                if (fbo == GL_INVALID_INDEX) {
                    this->fbo = glRequestFBO();
                }

                if (tex == GL_INVALID_INDEX) {
                    this->tex = glRequestTex();
                }

                this->resize(this->size, true);
                return true;
            }

            bool Texture::cleanup() {
                unbind();
                glReleaseFBO(&fbo);
                glReleaseTex(&tex);
                return true;
            }

            void Texture::resize(const Vec2& size, bool force) {
                glCheckError(name);

                if (this->size == size && !force) {
                    return;
                }

                this->size = size;

                verify();
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

                fill(cv::Mat::zeros(size.y(), size.x(), CV_8UC4));
                glCheckError(name);
            }

            void Texture::verify() const {
                if (size.isZero()) {
                    throw std::runtime_error(
                        "Texture::Module => constructed with zero size");
                }

                switch (type) {
                    case GL_TEXTURE_2D:
                    case GL_TEXTURE_3D:
                    case GL_TEXTURE_2D_ARRAY:
                    case GL_TEXTURE_CUBE_MAP: break;
                    default:
                        throw std::runtime_error(
                            "Texture::Module => unsupported texture type");
                }

                switch (intern) {
                    case GL_UNSIGNED_BYTE:
                    case GL_HALF_FLOAT:
                    case GL_FLOAT: break;
                    default:
                        throw std::runtime_error(
                            "Texture::Module => unsupported texture format");
                }
            }

            void Texture::bind() const {
                glCheckError(name);
                glActiveTexture(unit);
                glBindTexture(type, tex);
                glTexParameteri(type, GL_TEXTURE_WRAP_S, wrap);
                glTexParameteri(type, GL_TEXTURE_WRAP_T, wrap);
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

            void Texture::fill(cv::Mat image, uint layer, FillMode mode) {
                if (image.empty())
                    throw std::runtime_error(
                        "Texture::fill => empty image provided");

                if (depth <= layer)
                    throw std::runtime_error(
                        "Texture::fill => layer index out of bounds");

                glCheckError(name);
                const Vec2 imsize(image.size.p[1], image.size.p[0]);

                switch (mode) {
                    case RESIZE_TEXTURE: {
                        if (this->size != imsize) {
                            this->resize(imsize);
                        }

                        break;
                    }

                    case RESIZE_IMAGE: {
                        if (this->size != imsize) {
                            cv::resize(
                                image,
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
                glPixelStorei(GL_UNPACK_ROW_LENGTH,
                              image.step / image.elemSize());

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

            void Texture::copyTo(const Ptr& target,
                                 GLenum     mask,
                                 GLuint     filter) const {
                glCheckError(name);

                if (this->type != target->type) {
                    throw std::runtime_error(
                        "Texture::copyTo => type mismatch");
                }

                if (this->intern != target->intern) {
                    throw std::runtime_error(
                        "Texture::copyTo => format mismatch");
                }

                target->resize(this->size);

                glBindFramebuffer(GL_READ_FRAMEBUFFER, this->fbo);
                glFramebufferTexture2D(GL_READ_FRAMEBUFFER,
                                       GL_COLOR_ATTACHMENT0,
                                       instances.front(),
                                       this->tex,
                                       0);
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target->fbo);
                glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
                                       GL_COLOR_ATTACHMENT0,
                                       target->instances.front(),
                                       target->tex,
                                       0);

                glBlitFramebuffer(0,
                                  0,
                                  this->size.x(),
                                  this->size.y(),
                                  0,
                                  0,
                                  target->size.x(),
                                  target->size.y(),
                                  mask,
                                  filter);

                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glGenerateMipmap(target->type);
                target->unbind();

                glCheckError(name);
            }

        };  // namespace opengl
    };      // namespace render
};          // namespace corekit