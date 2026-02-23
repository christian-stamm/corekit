#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>

#include <vector>

#include "corekit/cuda/core.hpp"
#include "corekit/cuda/image.hpp"

namespace corekit {
    namespace cuda {

        constexpr uint MAX_TEXT_LEN  = 64;
        constexpr uint DEF_TEXT_SIZE = 32;

        struct GlyphInfo {
            int x;
            int y;
            int w;
            int h;
            int advance;
            int bearingX;
            int bearingY;
        };

        struct Font {
            uint             font_size;
            uint2            atlas_size;
            uint             glyphCount;
            uint             ascent;
            uint             descent;
            uint             lineGap;
            NvMem<uint>      d_atlas;   // device alpha atlas
            NvMem<GlyphInfo> d_glyphs;  // device glyph metrics
            uint*            d_atlas_ptr;
            GlyphInfo*       d_glyphs_ptr;

            static Font* loadFont(const char* fontPath,
                                  uint        size = DEF_TEXT_SIZE);

            static void freeFont(Font* font);
        };

        struct Text {
            using List = std::vector<Text>;

            char   msg[MAX_TEXT_LEN];  // null-terminated text string
            uint2  pos;                // (x, y) position of the text
            uchar4 color;              // (r, g, b, a)
        };

        struct Line {
            using List = std::vector<Line>;

            uint2  origin;     // (x, y) position of the line start point
            uint2  target;     // (x, y) position of the line end point
            uchar4 p0_col;     // (r, g, b, a)
            uchar4 p1_col;     // (r, g, b, a)
            uint   thickness;  // line thickness
        };

        struct Rect {
            using List = std::vector<Rect>;

            uint2  center;     // (cx, cy)
            uint2  shape;      // (w, h)
            uchar4 border;     // (r, g, b, a)
            uchar4 fill;       // (r, g, b, a)
            uint   thickness;  // border thickness
        };

        struct Circle {
            using List = std::vector<Circle>;

            uint2  center;     // (cx, cy) center of the circle
            uint   radius;     // radius of the circle
            uchar4 border;     // (r, g, b, a)
            uchar4 fill;       // (r, g, b, a)
            uint   thickness;  // border thickness
        };

        void drawText(Image3U&          img,
                      const Text::List& objs,
                      const Font*       font,
                      cudaStream_t      stream = 0);
        void drawText(Image3U&     img,
                      Text*        d_text,
                      uint         count,
                      const Font*  font,
                      cudaStream_t stream = 0);

        void drawLine(Image3U&          img,
                      const Line::List& objs,
                      cudaStream_t      stream = 0);
        void drawLine(Image3U&     img,
                      Line*        d_lines,
                      uint         count,
                      cudaStream_t stream = 0);

        void drawRect(Image3U&          img,
                      const Rect::List& objs,
                      cudaStream_t      stream = 0);
        void drawRect(Image3U&     img,
                      const Rect*  d_rects,
                      uint         count,
                      cudaStream_t stream = 0);

        void drawCircle(Image3U&            img,
                        const Circle::List& objs,
                        cudaStream_t        stream = 0);
        void drawCircle(Image3U&     img,
                        Circle*      d_circles,
                        uint         count,
                        cudaStream_t stream = 0);

        void overlay(Image3U&       img,
                     const Image4U& mask,
                     uint2          center,
                     cudaStream_t   stream = 0);

    }  // namespace cuda
}  // namespace corekit
