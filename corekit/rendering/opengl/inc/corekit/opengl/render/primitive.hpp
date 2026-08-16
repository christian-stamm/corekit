// #pragma once

// #include <ft2build.h>

// #include <algorithm>
// #include <cstdint>
// #include <glm/glm.hpp>
// #include <map>
// #include <stdexcept>
// #include <string>
// #include <unordered_map>
// #include <vector>

// #include "corekit/types.hpp"

// // KEEP GLAD BEFORE GLFW
// #include <glad/glad.h>
// // KEEP GLAD BEFORE GLFW
// #include <GLFW/glfw3.h>
// // KEEP GLAD BEFORE GLFW
// #include FT_FREETYPE_H

// namespace corekit {
//     namespace opengl {

//         using namespace corekit::types;

//         struct Primitive {
//             using List = std::vector<Primitive>;

//             enum Type {
//                 TEXT   = 1,
//                 LINE   = 2,
//                 RECT   = 3,
//                 CIRCLE = 4,
//             };

//             static Primitive::List& fuse(List& base, const List& extension) {
//                 base.insert(base.end(), extension.begin(), extension.end());
//                 return base;
//             }

//             Type type;

//             glm::vec4 coordA = glm::vec4(0.0f);
//             glm::vec4 coordB = glm::vec4(0.0f);

//             glm::vec4 colorA = glm::vec4(0.0f);
//             glm::vec4 colorB = glm::vec4(0.0f);

//             float fparamA = 0.0f;
//             float fparamB = 0.0f;
//         };

//         struct Line {
//             using List = std::vector<Line>;

//             static Primitive::List parse(const List& lines) {
//                 Primitive::List ps;
//                 ps.reserve(lines.size());
//                 for (const Line& r : lines) {
//                     ps.push_back(r.parse());
//                 }
//                 return ps;
//             }

//             Primitive parse() const {
//                 Primitive p;
//                 p.type   = Primitive::LINE;
//                 p.coordA = glm::vec4(origin, target);
//                 p.coordB = glm::vec4(thickness, border, feather, 0.0f);
//                 p.colorA = outerColor;
//                 p.colorB = innerColor;
//                 return p;
//             }

//             glm::vec2 origin     = glm::vec2(0.0f);
//             glm::vec2 target     = glm::vec2(0.0f);
//             glm::vec4 outerColor = glm::vec4(1.0f);
//             glm::vec4 innerColor = glm::vec4(1.0f);
//             float     thickness  = 2.0f;
//             float     border     = 0.0f;
//             float     feather    = 1.0f;
//         };

//         struct Rect {
//             using List = std::vector<Rect>;

//             static Primitive::List parse(const List& rects) {
//                 Primitive::List ps;
//                 ps.reserve(rects.size());
//                 for (const Rect& r : rects) {
//                     ps.push_back(r.parse());
//                 }
//                 return ps;
//             }

//             Primitive parse() const {
//                 Primitive p;
//                 p.type    = Primitive::RECT;
//                 p.coordA  = glm::vec4(pos, dim);
//                 p.colorA  = outerColor;
//                 p.colorB  = innerColor;
//                 p.fparamA = border;
//                 return p;
//             }

//             glm::vec2 pos        = glm::vec2(0.0f);
//             glm::vec2 dim        = glm::vec2(0.0f);
//             float     border     = 0.0f;
//             glm::vec4 outerColor = glm::vec4(1.0f);
//             glm::vec4 innerColor = glm::vec4(1.0f);
//         };

//         struct Circle {
//             using List = std::vector<Circle>;

//             static Primitive::List parse(const List& circles) {
//                 Primitive::List ps;
//                 ps.reserve(circles.size());
//                 for (const Circle& c : circles) {
//                     ps.push_back(c.parse());
//                 }
//                 return ps;
//             }

//             Primitive parse() const {
//                 Primitive p;
//                 p.type   = Primitive::CIRCLE;
//                 p.coordA = glm::vec4(center, radius, 0.0f);
//                 p.coordB = glm::vec4(border, feather, 0.0f, 0.0f);
//                 p.colorA = outerColor;
//                 p.colorB = innerColor;
//                 return p;
//             }

//             glm::vec2 center     = glm::vec2(0.0f);
//             float     radius     = 0.0f;
//             float     border     = 0.0f;
//             float     feather    = 1.0f;
//             glm::vec4 outerColor = glm::vec4(1.0f);
//             glm::vec4 innerColor = glm::vec4(1.0f);
//         };

//         struct Glyph {
//             using Atlas = std::unordered_map<char, Glyph>;

//             static constexpr int kFirstChar  = 32;
//             static constexpr int kLastChar   = 126;
//             static constexpr int kGlyphCount = (kLastChar - kFirstChar + 1);

//             struct AtlasData;

//             float     advance  = 0.0f;
//             glm::vec2 size     = glm::vec2(0.0f);
//             glm::vec2 bearing  = glm::vec2(0.0f);
//             glm::vec4 uvBounds = glm::vec4(0.0f);

//             static AtlasData loadAtlas(const Path& file, uint size);
//         };

//         struct Glyph::AtlasData {
//             Glyph::Atlas         glyphs;
//             glm::ivec2           atlasSize = glm::ivec2(0);
//             std::vector<uint8_t> bitmap;
//             uint                 ascent   = 0;
//             uint                 descent  = 0;
//             uint                 lineGap  = 0;
//             uint                 fontSize = 0;
//         };

//         struct Text {
//             using List = std::vector<Text>;

//             enum class Origin {
//                 TopLeft,
//                 Baseline,
//             };

//             std::string msg    = std::string();
//             glm::vec2   pos    = glm::vec2(0.0f);
//             glm::vec4   color  = glm::vec4(1.0f);
//             float       size   = 16.0f;
//             Origin      origin = Origin::TopLeft;
//         };

//         inline Glyph::AtlasData Glyph::loadAtlas(const Path& file, uint size)
//         {
//             AtlasData data;
//             data.fontSize = size;

//             if (file.empty()) {
//                 throw std::runtime_error(
//                     "Glyph::loadAtlas => empty font file path");
//             }

//             FT_Library ft   = nullptr;
//             FT_Face    face = nullptr;

//             const std::string fontPath = file.string();

//             if (FT_Init_FreeType(&ft) != 0) {
//                 throw std::runtime_error(
//                     "Glyph::loadAtlas => failed to initialize FreeType");
//             }

//             if (FT_New_Face(ft, fontPath.c_str(), 0, &face) != 0) {
//                 FT_Done_FreeType(ft);
//                 throw std::runtime_error(
//                     "Glyph::loadAtlas => failed to load font face");
//             }

//             FT_Set_Pixel_Sizes(face, 0, size);

//             int ascent  = static_cast<int>(face->size->metrics.ascender /
//             64); int descent =
//             static_cast<int>(-face->size->metrics.descender / 64); int height
//             = static_cast<int>(face->size->metrics.height / 64); int lineGap
//             = height - ascent - descent;

//             ascent  = std::max(ascent, 0);
//             descent = std::max(descent, 0);
//             lineGap = std::max(lineGap, 0);

//             int cellW = 0;
//             int cellH = 0;

//             for (int c = kFirstChar; c <= kLastChar; ++c) {
//                 if (FT_Load_Char(face, c, FT_LOAD_RENDER) != 0) {
//                     continue;
//                 }

//                 const FT_GlyphSlot g = face->glyph;
//                 cellW = std::max(cellW, static_cast<int>(g->bitmap.width));
//                 cellH = std::max(cellH, static_cast<int>(g->bitmap.rows));
//             }

//             if (cellW == 0 || cellH == 0) {
//                 FT_Done_Face(face);
//                 FT_Done_FreeType(ft);
//                 throw std::runtime_error(
//                     "Glyph::loadAtlas => failed to determine atlas cell "
//                     "size");
//             }

//             const int atlasW = cellW * kGlyphCount;
//             const int atlasH = cellH;

//             std::vector<uint8_t> atlas(
//                 static_cast<size_t>(atlasW) * static_cast<size_t>(atlasH),
//                 0u);

//             for (int c = kFirstChar; c <= kLastChar; ++c) {
//                 if (FT_Load_Char(face, c, FT_LOAD_RENDER) != 0) {
//                     continue;
//                 }

//                 const FT_GlyphSlot g       = face->glyph;
//                 const int          idx     = c - kFirstChar;
//                 const int          offsetX = idx * cellW;
//                 const int          offsetY = 0;

//                 for (int row = 0; row < static_cast<int>(g->bitmap.rows);
//                      ++row) {
//                     const int dstRow =
//                         offsetY + (static_cast<int>(g->bitmap.rows) - 1 -
//                         row);
//                     const unsigned char* src =
//                         g->bitmap.buffer + row * g->bitmap.pitch;
//                     uint8_t* dst = atlas.data() + dstRow * atlasW + offsetX;

//                     for (int col = 0; col <
//                     static_cast<int>(g->bitmap.width);
//                          ++col) {
//                         dst[col] = src[col];
//                     }
//                 }

//                 Glyph glyph;
//                 glyph.advance = static_cast<float>(g->advance.x) / 64.0f;
//                 glyph.size    =
//                 glm::vec2(static_cast<float>(g->bitmap.width),
//                                        static_cast<float>(g->bitmap.rows));
//                 glyph.bearing = glm::vec2(static_cast<float>(g->bitmap_left),
//                                           static_cast<float>(g->bitmap_top));

//                 const float u0 =
//                     static_cast<float>(offsetX) / static_cast<float>(atlasW);
//                 const float v0 =
//                     static_cast<float>(offsetY) / static_cast<float>(atlasH);
//                 glyph.uvBounds =
//                     glm::vec4(u0,
//                               v0,
//                               glyph.size.x / static_cast<float>(atlasW),
//                               glyph.size.y / static_cast<float>(atlasH));

//                 data.glyphs.emplace(static_cast<char>(c), glyph);
//             }

//             data.atlasSize = glm::ivec2(atlasW, atlasH);
//             data.bitmap    = std::move(atlas);
//             data.ascent    = static_cast<uint>(ascent);
//             data.descent   = static_cast<uint>(descent);
//             data.lineGap   = static_cast<uint>(lineGap);

//             FT_Done_Face(face);
//             FT_Done_FreeType(ft);

//             return data;
//         }

//     };  // namespace opengl
// };      // namespace corekit
