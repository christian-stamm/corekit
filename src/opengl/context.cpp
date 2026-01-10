#include "corekit/opengl/context.hpp"

#include <format>
#include <string>

#include "corekit/core.hpp"

namespace corekit {
    namespace opengl {

        using namespace corekit::system;

        void glCheckError(std::string                 meta,
                          const std::source_location& location) {
            const GLenum& errcode = glGetError();

            corecheck(errcode == GL_NO_ERROR,
                      std::format("OpenGL Code: 0x{:04X} ({:08}) {}",
                                  errcode,
                                  errcode,
                                  meta),
                      location);
        }

        GLuint glRequestPrg() {
            return glCreateProgram();
        }

        GLuint glRequestShd(GLenum type) {
            return glCreateShader(type);
        }

        GLuint glRequestFbo() {
            GLuint fbo;
            glGenFramebuffers(1, &fbo);
            return fbo;
        }

        GLuint glRequestTex() {
            GLuint tex;
            glGenTextures(1, &tex);
            return tex;
        }

        GLuint glRequestVBO() {
            GLuint vbo;
            glGenBuffers(1, &vbo);
            return vbo;
        }

        GLuint glRequestVAO() {
            GLuint vao;
            glGenVertexArrays(1, &vao);
            return vao;
        }

        void glReleaseFbo(const GLuint* fbo) {
            glDeleteFramebuffers(1, fbo);
        }

        void glReleasePrg(const GLuint* prg) {
            glDeleteProgram(*prg);
        }

        void glReleaseShd(const GLuint* shd) {
            glDeleteShader(*shd);
        }

        void glReleaseTex(const GLuint* tex) {
            glDeleteTextures(1, tex);
        }

        void glReleaseVBO(const GLuint* vbo) {
            glDeleteBuffers(1, vbo);
        }

        void glReleaseVAO(const GLuint* vao) {
            glDeleteVertexArrays(1, vao);
        }

    };  // namespace opengl
};  // namespace corekit