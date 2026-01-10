#pragma once
// KEEP GLAD BEFORE GLFW
#include <glad/glad.h>
// KEEP GLAD BEFORE GLFW
#include <GLFW/glfw3.h>
// KEEP GLAD BEFORE GLFW

#include <glm/glm.hpp>
#include <source_location>
#include <string>

namespace corekit {
    namespace opengl {

        void glCheckError(std::string                 meta = "",
                          const std::source_location& location =
                              std::source_location::current());

        GLuint glRequestPrg();
        GLuint glRequestShd(GLenum type);
        GLuint glRequestFbo();
        GLuint glRequestTex();
        GLuint glRequestVBO();
        GLuint glRequestVAO();

        void glReleasePrg(const GLuint* prg);
        void glReleaseShd(const GLuint* shd);
        void glReleaseFbo(const GLuint* fbo);
        void glReleaseTex(const GLuint* tex);
        void glReleaseVBO(const GLuint* vbo);
        void glReleaseVAO(const GLuint* vao);

    };  // namespace opengl
};  // namespace corekit
