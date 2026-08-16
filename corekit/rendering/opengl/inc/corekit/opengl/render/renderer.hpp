// #pragma once
// #include <array>
// #include <exception>
// #include <memory>
// #include <stdexcept>
// #include <type_traits>
// #include <vector>

// #include "corekit/opengl/program.hpp"
// #include "corekit/opengl/render/primitive.hpp"
// #include "corekit/opengl/texture.hpp"
// #include "corekit/utils/assert.hpp"
// #include "corekit/utils/filemgr.hpp"

// namespace corekit {
//     namespace opengl {

//         using namespace corekit::system;
//         using namespace corekit::types;
//         using namespace corekit::utils;

//         class Renderer : public Program {
//            public:
//             using Ptr = std::shared_ptr<Renderer>;

//             struct Settings {
//                 Path shaderDir = Path();
//                 Path fontDir   = Path();
//                 Path fontFile  = Path();
//                 int  fontSize  = 32;
//             };

//             Renderer(const Settings& cfg)
//                 : Program(buildProgram(cfg))
//                 , cfg(cfg)
//                 , canvas(nullptr)
//                 , vao(0)
//                 , vbo_quad(0)
//                 , vbo_instance(0) {}

//             void draw(const Line::List& objs) {
//                 primitives = Primitive::fuse(primitives, Line::parse(objs));
//             }

//             void draw(const Rect::List& objs) {
//                 primitives = Primitive::fuse(primitives, Rect::parse(objs));
//             }

//             void draw(const Circle::List& objs) {
//                 primitives = Primitive::fuse(primitives,
//                 Circle::parse(objs));
//             }

//             void draw(const Text::List& objs) {
//                 primitives = Primitive::fuse(primitives, parseText(objs));
//             }

//             void reconfigure(const Texture::Ptr canvas) {
//                 this->canvas = canvas;
//                 this->primitives.clear();
//             }

//            protected:
//             static Program::Settings buildProgram(const Settings& cfg) {
//                 return Program::Settings{
//                     .hash = "PrimRend",
//                     .shaders =
//                         {
//                             Shader::Settings{
//                                 .hash = "Vert PrimRend",
//                                 .code = File::loadTxt(cfg.shaderDir /
//                                                       "prim_overlay.vert"),
//                                 .type = GL_VERTEX_SHADER,
//                                 .glID = GL_INVALID_INDEX,
//                             },
//                             Shader::Settings{
//                                 .hash = "Frag PrimRend",
//                                 .code = File::loadTxt(cfg.shaderDir /
//                                                       "prim_overlay.frag"),
//                                 .type = GL_FRAGMENT_SHADER,
//                                 .glID = GL_INVALID_INDEX,
//                             },
//                         },
//                 };
//             }

//             bool prepare() override {
//                 // Quad vertices for a unit quad (for instancing)
//                 const float quad[] =
//                     {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
//                 const size_t stride = sizeof(Primitive);

//                 vao          = glRequestVAO();
//                 vbo_quad     = glRequestVBO();
//                 vbo_instance = glRequestVBO();

//                 glBindVertexArray(vao);

//                 // Setup Quad Buffer (per-vertex)
//                 glBindBuffer(GL_ARRAY_BUFFER, vbo_quad);
//                 glBufferData(GL_ARRAY_BUFFER,
//                              sizeof(quad),
//                              quad,
//                              GL_STATIC_DRAW);
//                 glEnableVertexAttribArray(0);
//                 glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
//                 glVertexAttribDivisor(0, 0);  // Per-vertex

//                 // Setup Instance Buffer (per-instance attributes)
//                 glBindBuffer(GL_ARRAY_BUFFER, vbo_instance);
//                 setupAttrInt(1,
//                              1,
//                              GL_INT,
//                              stride,
//                              offsetof(Primitive, type));  // inType
//                 setupAttr(2,
//                           4,
//                           GL_FLOAT,
//                           stride,
//                           offsetof(Primitive, coordA));  // inCoordA
//                 setupAttr(3,
//                           4,
//                           GL_FLOAT,
//                           stride,
//                           offsetof(Primitive, coordB));  // inCoordB
//                 setupAttr(4,
//                           4,
//                           GL_FLOAT,
//                           stride,
//                           offsetof(Primitive, colorA));  // inColorA
//                 setupAttr(5,
//                           4,
//                           GL_FLOAT,
//                           stride,
//                           offsetof(Primitive, colorB));  // inColorB
//                 setupAttr(6,
//                           1,
//                           GL_FLOAT,
//                           stride,
//                           offsetof(Primitive, fparamA));  // inFparamA
//                 setupAttr(7,
//                           1,
//                           GL_FLOAT,
//                           stride,
//                           offsetof(Primitive, fparamB));  // inFparamB
//                 glBindBuffer(GL_ARRAY_BUFFER, 0);

//                 glBindVertexArray(0);
//                 bool ready = Program::prepare();
//                 if (ready) {
//                     initFont();
//                 }
//                 return ready;
//             }

//             bool cleanup() override {
//                 glReleaseVAO(&vao);
//                 glReleaseVBO(&vbo_quad);
//                 glReleaseVBO(&vbo_instance);
//                 releaseFont();

//                 reconfigure(nullptr);
//                 return Program::cleanup();
//             }

//             void render() const override {
//                 corecheck(canvas != nullptr, "No Canvas provided");

//                 const Vec2 size = canvas->size;

//                 glBindFramebuffer(GL_FRAMEBUFFER, canvas->fbo);
//                 glFramebufferTexture(GL_FRAMEBUFFER,
//                                      GL_COLOR_ATTACHMENT0,
//                                      canvas->tex,
//                                      0);

//                 glViewport(0, 0, size.x(), size.y());

//                 const GLboolean blendWasEnabled = glIsEnabled(GL_BLEND);
//                 glEnable(GL_BLEND);
//                 glBlendEquation(GL_FUNC_ADD);
//                 glBlendFuncSeparate(GL_SRC_ALPHA,
//                                     GL_ONE_MINUS_SRC_ALPHA,
//                                     GL_ONE,
//                                     GL_ONE_MINUS_SRC_ALPHA);

//                 canvas->bind();

//                 const bool hasFont = font.ready();
//                 glUniform1i(getUniform("pluginTex"), 0);
//                 glUniform1i(getUniform("fontReady"), hasFont ? 1 : 0);
//                 if (hasFont) {
//                     font.atlas->bind();
//                     glUniform1i(getUniform("fontAtlas"),
//                                 static_cast<GLint>(font.atlas->getSlot()));
//                 }

//                 glBindVertexArray(vao);
//                 glBindBuffer(GL_ARRAY_BUFFER, vbo_instance);
//                 glBufferData(GL_ARRAY_BUFFER,
//                              primitives.size() * sizeof(Primitive),
//                              primitives.data(),
//                              GL_DYNAMIC_DRAW);

//                 drawPrimitives();

//                 glBindBuffer(GL_ARRAY_BUFFER, 0);
//                 glBindVertexArray(0);

//                 if (hasFont) {
//                     font.atlas->unbind();
//                 }
//                 canvas->unbind();

//                 if (blendWasEnabled == GL_FALSE) {
//                     glDisable(GL_BLEND);
//                 }

//                 glBindFramebuffer(GL_FRAMEBUFFER, 0);

//                 primitives.clear();
//             }

//            private:
//             struct Font {
//                 Texture::Ptr atlas;
//                 glm::ivec2   size = glm::ivec2(0);
//                 Glyph::Atlas glyphs;
//                 uint         fontSize = 0;
//                 uint         ascent   = 0;
//                 uint         descent  = 0;
//                 uint         lineGap  = 0;

//                 bool ready() const {
//                     return atlas && atlas->isLoaded() && !glyphs.empty() &&
//                            size.x > 0 && size.y > 0;
//                 }
//             };

//             Primitive::List parseText(const Text::List& objs) const;
//             void            initFont();
//             void            releaseFont();
//             Path            resolveFontPath() const;
//             void            drawPrimitives() const;

//             void setupAttr(int    loc,
//                            int    size,
//                            GLenum type,
//                            int    stride,
//                            size_t offset) {
//                 glEnableVertexAttribArray(loc);
//                 glVertexAttribPointer(loc,
//                                       size,
//                                       type,
//                                       GL_FALSE,
//                                       stride,
//                                       (void*)(offset));
//                 glVertexAttribDivisor(loc, 1);
//             }

//             void setupAttrInt(int    loc,
//                               int    size,
//                               GLenum type,
//                               int    stride,
//                               size_t offset) {
//                 glEnableVertexAttribArray(loc);
//                 glVertexAttribIPointer(loc,
//                                        size,
//                                        type,
//                                        stride,
//                                        (void*)(offset));
//                 glVertexAttribDivisor(loc, 1);
//             }

//             Settings cfg;
//             Font     font;

//             GLuint vao;
//             GLuint vbo_quad;
//             GLuint vbo_instance;

//             mutable Texture::Ptr    canvas;
//             mutable Primitive::List primitives;
//         };

//         inline void Renderer::drawPrimitives() const {
//             if (primitives.empty()) {
//                 return;
//             }

//             glDrawArraysInstanced(GL_TRIANGLE_STRIP,
//                                   0,
//                                   4,
//                                   static_cast<GLsizei>(primitives.size()));
//         }

//         inline Primitive::List Renderer::parseText(
//             const Text::List& objs) const {
//             Primitive::List ps;
//             if (!font.ready() || objs.empty()) {
//                 return ps;
//             }

//             const float baseSize =
//                 font.fontSize > 0 ? static_cast<float>(font.fontSize) : 1.0f;
//             const float lineStep =
//                 static_cast<float>(font.ascent + font.descent +
//                 font.lineGap);
//             const float lineDelta = lineStep > 0.0f ? lineStep : baseSize;

//             for (const Text& txt : objs) {
//                 const float targetSize = txt.size > 0.0f ? txt.size :
//                 baseSize; const float scale      = targetSize / baseSize;
//                 const bool  originTop  = (txt.origin ==
//                 Text::Origin::TopLeft); glm::vec2   cursor(
//                     txt.pos.x,
//                     originTop ? txt.pos.y + font.ascent * scale : txt.pos.y);

//                 for (const char c : txt.msg) {
//                     if (c == '\n') {
//                         cursor.x = txt.pos.x;
//                         cursor.y +=
//                             (originTop ? 1.0f : -1.0f) * lineDelta * scale;
//                         continue;
//                     }

//                     const auto glyphIt = font.glyphs.find(c);
//                     if (glyphIt == font.glyphs.end()) {
//                         continue;
//                     }

//                     const Glyph& glyph = glyphIt->second;
//                     if (glyph.size.x <= 0.0f || glyph.size.y <= 0.0f) {
//                         cursor.x += glyph.advance * scale;
//                         continue;
//                     }

//                     const float x = cursor.x + glyph.bearing.x * scale;
//                     const float y = cursor.y - glyph.bearing.y * scale;
//                     const float w = glyph.size.x * scale;
//                     const float h = glyph.size.y * scale;

//                     Primitive p;
//                     p.type   = Primitive::TEXT;
//                     p.coordA = glm::vec4(x, y, w, h);
//                     p.coordB = glyph.uvBounds;
//                     p.colorA = txt.color;
//                     ps.push_back(p);

//                     cursor.x += glyph.advance * scale;
//                 }
//             }

//             return ps;
//         }

//         inline void Renderer::initFont() {
//             releaseFont();

//             const Path fontPath = resolveFontPath();
//             if (fontPath.empty()) {
//                 logger(Level::WARN)
//                     << "Renderer => no font file found, text disabled";
//                 return;
//             }

//             const uint pixelSize =
//                 cfg.fontSize > 0 ? static_cast<uint>(cfg.fontSize) : 32u;

//             try {
//                 auto atlas    = Glyph::loadAtlas(fontPath, pixelSize);
//                 font.fontSize = atlas.fontSize;
//                 font.ascent   = atlas.ascent;
//                 font.descent  = atlas.descent;
//                 font.lineGap  = atlas.lineGap;
//                 font.size     = atlas.atlasSize;
//                 font.glyphs   = std::move(atlas.glyphs);

//                 if (font.size.x <= 0 || font.size.y <= 0 ||
//                     atlas.bitmap.empty()) {
//                     releaseFont();
//                     logger(Level::WARN)
//                         << "Renderer => empty font atlas for " << fontPath;
//                     return;
//                 }

//                 Texture::Settings texCfg;
//                 texCfg.hash   = "FontAtlas";
//                 texCfg.size   = Vec2(static_cast<float>(font.size.x),
//                                    static_cast<float>(font.size.y));
//                 texCfg.type   = GL_TEXTURE_2D;
//                 texCfg.unit   = GL_TEXTURE1;
//                 texCfg.wrap   = GL_CLAMP_TO_EDGE;
//                 texCfg.filter = Texture::Filter{GL_LINEAR, GL_LINEAR};

//                 font.atlas = Texture::build(texCfg);
//                 if (!font.atlas->load()) {
//                     throw std::runtime_error(
//                         "Renderer => failed to initialize font atlas");
//                 }

//                 const size_t pixelCount = static_cast<size_t>(font.size.x) *
//                                           static_cast<size_t>(font.size.y);
//                 if (atlas.bitmap.size() != pixelCount) {
//                     throw std::runtime_error(
//                         "Renderer => font atlas bitmap size mismatch");
//                 }

//                 std::vector<uint8_t> rgba(pixelCount * 4u, 255u);
//                 for (size_t idx = 0; idx < pixelCount; ++idx) {
//                     rgba[idx * 4u + 3u] = atlas.bitmap[idx];
//                 }

//                 font.atlas->bind();
//                 glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//                 glTexSubImage2D(GL_TEXTURE_2D,
//                                 0,
//                                 0,
//                                 0,
//                                 font.size.x,
//                                 font.size.y,
//                                 GL_RGBA,
//                                 GL_UNSIGNED_BYTE,
//                                 rgba.data());
//                 font.atlas->unbind();

//                 logger() << "Renderer => loaded font " << fontPath;
//             } catch (const std::exception& e) {
//                 logger(Level::WARN)
//                     << "Renderer => failed to load font: " << e.what();
//                 releaseFont();
//             }
//         }

//         inline void Renderer::releaseFont() {
//             if (font.atlas) {
//                 font.atlas->unload();
//                 font.atlas.reset();
//             }
//             font.size     = glm::ivec2(0);
//             font.fontSize = 0;
//             font.ascent   = 0;
//             font.descent  = 0;
//             font.lineGap  = 0;
//             font.glyphs.clear();
//         }

//         inline Path Renderer::resolveFontPath() const {
//             if (!cfg.fontFile.empty() && File::exists(cfg.fontFile)) {
//                 return cfg.fontFile;
//             }

//             if (!cfg.fontDir.empty() && File::exists(cfg.fontDir)) {
//                 const auto ttf = File::scan(cfg.fontDir, ".ttf");
//                 if (!ttf.empty()) {
//                     return ttf.front();
//                 }
//                 const auto otf = File::scan(cfg.fontDir, ".otf");
//                 if (!otf.empty()) {
//                     return otf.front();
//                 }
//             }

//             static const std::array<Path, 3> defaults = {
//                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
//                 Path("/usr/share/fonts/truetype/liberation/"
//                      "LiberationSans-Regular.ttf"),
//                 Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
//             };

//             for (const auto& candidate : defaults) {
//                 if (File::exists(candidate)) {
//                     return candidate;
//                 }
//             }

//             return Path();
//         }

//     }  // namespace opengl
// }  // namespace corekit
