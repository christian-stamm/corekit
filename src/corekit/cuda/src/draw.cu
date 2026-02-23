#include <cuda_runtime.h>
#include <ft2build.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "corekit/cuda/core.hpp"
#include "corekit/cuda/draw.hpp"
#include "corekit/cuda/image.hpp"
#include FT_FREETYPE_H

namespace corekit {
    namespace cuda {

        constexpr int kFirstChar  = 32;
        constexpr int kLastChar   = 126;
        constexpr int kGlyphCount = (kLastChar - kFirstChar + 1);
        constexpr int kMaxTextLen = MAX_TEXT_LEN;

        struct GlyphInstance {
            int    glyphIndex;
            int    dstX;
            int    dstY;
            uchar4 color;
        };

        Font* Font::loadFont(const char* fontPath, uint size) {
            if (fontPath == nullptr) {
                return nullptr;
            }

            FT_Library ft   = nullptr;
            FT_Face    face = nullptr;

            if (FT_Init_FreeType(&ft) != 0) {
                return nullptr;
            }

            if (FT_New_Face(ft, fontPath, 0, &face) != 0) {
                FT_Done_FreeType(ft);
                return nullptr;
            }

            FT_Set_Pixel_Sizes(face, 0, size);

            int ascent  = static_cast<int>(face->size->metrics.ascender / 64);
            int descent = static_cast<int>(-face->size->metrics.descender / 64);
            int height  = static_cast<int>(face->size->metrics.height / 64);
            int lineGap = height - ascent - descent;

            if (ascent < 0) {
                ascent = 0;
            }
            if (descent < 0) {
                descent = 0;
            }
            if (lineGap < 0) {
                lineGap = 0;
            }

            int cellW = 0;
            int cellH = 0;

            std::vector<GlyphInfo> glyphs(kGlyphCount);

            for (int c = kFirstChar; c <= kLastChar; ++c) {
                if (FT_Load_Char(face, c, FT_LOAD_RENDER) != 0) {
                    continue;
                }

                const int          idx = c - kFirstChar;
                const FT_GlyphSlot g   = face->glyph;

                cellW = std::max(cellW, static_cast<int>(g->bitmap.width));
                cellH = std::max(cellH, static_cast<int>(g->bitmap.rows));

                glyphs[idx].w        = static_cast<int>(g->bitmap.width);
                glyphs[idx].h        = static_cast<int>(g->bitmap.rows);
                glyphs[idx].advance  = static_cast<int>(g->advance.x / 64);
                glyphs[idx].bearingX = static_cast<int>(g->bitmap_left);
                glyphs[idx].bearingY = static_cast<int>(g->bitmap_top);
            }

            if (cellW == 0 || cellH == 0) {
                FT_Done_Face(face);
                FT_Done_FreeType(ft);
                return nullptr;
            }

            const int atlasW = cellW * kGlyphCount;
            const int atlasH = cellH;

            std::vector<uint> atlas(
                static_cast<size_t>(atlasW) * static_cast<size_t>(atlasH),
                0u);

            for (int c = kFirstChar; c <= kLastChar; ++c) {
                if (FT_Load_Char(face, c, FT_LOAD_RENDER) != 0) {
                    continue;
                }

                const int          idx     = c - kFirstChar;
                const FT_GlyphSlot g       = face->glyph;
                const int          offsetX = idx * cellW;
                const int          offsetY = 0;

                glyphs[idx].x = offsetX;
                glyphs[idx].y = offsetY;

                for (int row = 0; row < static_cast<int>(g->bitmap.rows);
                     ++row) {
                    const unsigned char* src =
                        g->bitmap.buffer + row * g->bitmap.pitch;
                    uint* dst =
                        atlas.data() + (offsetY + row) * atlasW + offsetX;

                    for (int col = 0; col < static_cast<int>(g->bitmap.width);
                         ++col) {
                        dst[col] = static_cast<uint>(src[col]);
                    }
                }
            }

            auto* font       = new Font{};
            font->font_size  = size;
            font->atlas_size = make_uint2(static_cast<uint>(atlasW),
                                          static_cast<uint>(atlasH));
            font->glyphCount = static_cast<uint>(kGlyphCount);
            font->ascent     = static_cast<uint>(ascent);
            font->descent    = static_cast<uint>(descent);
            font->lineGap    = static_cast<uint>(lineGap);
     
            font->d_atlas = NvMem<uint>( atlas.size());

            check_cuda(cudaMemcpy(font->d_atlas.ptr(),
                                  atlas.data(),
                                  font->d_atlas.get_bytes(),
                                  cudaMemcpyHostToDevice));

            font->d_glyphs = NvMem<GlyphInfo>(glyphs.size() );

            check_cuda(cudaMemcpy(font->d_glyphs.ptr(),
                                  glyphs.data(),
                                  font->d_glyphs.get_bytes(),
                                  cudaMemcpyHostToDevice));

            font->d_atlas_ptr  = font->d_atlas.ptr();
            font->d_glyphs_ptr = font->d_glyphs.ptr();

            FT_Done_Face(face);
            FT_Done_FreeType(ft);

            return font;
        }

        void Font::freeFont(Font* font) {
            if (!font) {
                return;
            }

            delete font;
        }

        namespace {

            __device__ __forceinline__ uchar3 blend_rgb(uchar3 dst,
                                                        uchar4 src,
                                                        float  alpha) {
                const float ia = 1.0f - alpha;
                return make_uchar3(
                    static_cast<unsigned char>(dst.x * ia + src.x * alpha),
                    static_cast<unsigned char>(dst.y * ia + src.y * alpha),
                    static_cast<unsigned char>(dst.z * ia + src.z * alpha));
            }

            __device__ __forceinline__ int iabs(int v) {
                return (v < 0) ? -v : v;
            }

            __device__ __forceinline__ int imax(int a, int b) {
                return (a > b) ? a : b;
            }

            __device__ __forceinline__ uchar4 lerp_rgba(uchar4 c0,
                                                        uchar4 c1,
                                                        float  t) {
                const float r =
                    static_cast<float>(c0.x) +
                    (static_cast<float>(c1.x) - static_cast<float>(c0.x)) * t;
                const float g =
                    static_cast<float>(c0.y) +
                    (static_cast<float>(c1.y) - static_cast<float>(c0.y)) * t;
                const float b =
                    static_cast<float>(c0.z) +
                    (static_cast<float>(c1.z) - static_cast<float>(c0.z)) * t;
                const float a =
                    static_cast<float>(c0.w) +
                    (static_cast<float>(c1.w) - static_cast<float>(c0.w)) * t;

                return make_uchar4(static_cast<unsigned char>(r + 0.5f),
                                   static_cast<unsigned char>(g + 0.5f),
                                   static_cast<unsigned char>(b + 0.5f),
                                   static_cast<unsigned char>(a + 0.5f));
            }

            __device__ __forceinline__ bool in_bounds(int   x,
                                                      int   y,
                                                      uint2 size) {
                return x >= 0 && y >= 0 && x < static_cast<int>(size.x) &&
                       y < static_cast<int>(size.y);
            }

            __device__ __forceinline__ void set_pixel(uchar*  img,
                                                      uint2   size,
                                                      int     x,
                                                      int     y,
                                                      uchar3  color) {
                if (!in_bounds(x, y, size)) {
                    return;
                }

                const int idx = (y * static_cast<int>(size.x) + x) * 3;
                img[idx + 0] = color.x;
                img[idx + 1] = color.y;
                img[idx + 2] = color.z;
            }

            __device__ __forceinline__ void blend_pixel(uchar*  img,
                                                        uint2   size,
                                                        int     x,
                                                        int     y,
                                                        uchar4  color) {
                if (!in_bounds(x, y, size)) {
                    return;
                }

                const int rgb_idx = (y * static_cast<int>(size.x) + x) * 3;
                const uchar3 current = make_uchar3(img[rgb_idx + 0], img[rgb_idx + 1], img[rgb_idx + 2]);
                const float alpha = static_cast<float>(color.w) / 255.0f;
                const uchar3 blended = blend_rgb(current, color, alpha);
                img[rgb_idx + 0] = blended.x;
                img[rgb_idx + 1] = blended.y;
                img[rgb_idx + 2] = blended.z;
            }

        }  // namespace

        __global__ void draw_lines_kernel(uchar*      img,
                                          uint2       img_size,
                                          const Line* lines,
                                          int         count) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= count) {
                return;
            }

            const Line line = lines[idx];

            if (line.thickness == 0 ||
                (line.p0_col.w == 0 && line.p1_col.w == 0)) {
                return;
            }

            const int x0 = static_cast<int>(line.origin.x);
            const int y0 = static_cast<int>(line.origin.y);
            const int x1 = static_cast<int>(line.target.x);
            const int y1 = static_cast<int>(line.target.y);

            const int dx     = x1 - x0;
            const int dy     = y1 - y0;
            const int steps  = imax(iabs(dx), iabs(dy));
            const int radius = imax(0, static_cast<int>(line.thickness) / 2);

            if (steps == 0) {
                const uchar4 color = lerp_rgba(line.p0_col, line.p1_col, 0.0f);
                for (int oy = -radius; oy <= radius; ++oy) {
                    for (int ox = -radius; ox <= radius; ++ox) {
                        blend_pixel(img, img_size, x0 + ox, y0 + oy, color);
                    }
                }
                return;
            }

            const float inv = 1.0f / static_cast<float>(steps);
            float       fx  = static_cast<float>(x0);
            float       fy  = static_cast<float>(y0);
            const float sx  = static_cast<float>(dx) * inv;
            const float sy  = static_cast<float>(dy) * inv;

            for (int i = 0; i <= steps; ++i) {
                const int    xi    = static_cast<int>(fx + 0.5f);
                const int    yi    = static_cast<int>(fy + 0.5f);
                const float  t     = static_cast<float>(i) * inv;
                const uchar4 color = lerp_rgba(line.p0_col, line.p1_col, t);

                for (int oy = -radius; oy <= radius; ++oy) {
                    for (int ox = -radius; ox <= radius; ++ox) {
                        blend_pixel(img, img_size, xi + ox, yi + oy, color);
                    }
                }

                fx += sx;
                fy += sy;
            }
        }

        __global__ void draw_rects_kernel(uchar*      img,
                                          uint2       img_size,
                                          const Rect* rects,
                                          int         count) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= static_cast<int>(img_size.x) ||
                y >= static_cast<int>(img_size.y)) {
                return;
            }

            const float px = static_cast<float>(x) + 0.5f;
            const float py = static_cast<float>(y) + 0.5f;

            const int idx = y * static_cast<int>(img_size.x) + x;
            const int rgb_idx = idx * 3;
            uchar3    out = {img[rgb_idx + 0], img[rgb_idx + 1], img[rgb_idx + 2]};

            for (int i = 0; i < count; ++i) {
                const Rect rect = rects[i];

                const float cx = static_cast<float>(rect.center.x);
                const float cy = static_cast<float>(rect.center.y);
                const float hw = static_cast<float>(rect.shape.x) * 0.5f;
                const float hh = static_cast<float>(rect.shape.y) * 0.5f;
                const float t  = static_cast<float>(rect.thickness);

                const float dx = fabsf(px - cx);
                const float dy = fabsf(py - cy);

                const float alphaFill =
                    static_cast<float>(rect.fill.w) / 255.0f;
                const float alphaBorder =
                    static_cast<float>(rect.border.w) / 255.0f;

                float innerW = hw - t * 0.5f;
                float innerH = hh - t * 0.5f;
                if (innerW < 0.0f) {
                    innerW = 0.0f;
                }
                if (innerH < 0.0f) {
                    innerH = 0.0f;
                }

                if (alphaFill > 0.0f && dx <= innerW && dy <= innerH) {
                    out = blend_rgb(out, rect.fill, alphaFill);
                    continue;
                }

                if (alphaBorder > 0.0f && dx <= hw && dy <= hh &&
                    (dx > innerW || dy > innerH)) {
                    out = blend_rgb(out, rect.border, alphaBorder);
                }
            }

            img[rgb_idx + 0] = out.x;
            img[rgb_idx + 1] = out.y;
            img[rgb_idx + 2] = out.z;
        }

        __global__ void draw_circles_kernel(uchar*        img,
                                            uint2         img_size,
                                            const Circle* circles,
                                            int           count) {
            const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx >= count) {
                return;
            }

            const Circle circle = circles[idx];

            if (circle.fill.w == 0 && circle.border.w == 0) {
                return;
            }

            const int cx = static_cast<int>(circle.center.x);
            const int cy = static_cast<int>(circle.center.y);
            const int r  = static_cast<int>(circle.radius);
            const int t  = static_cast<int>(circle.thickness);

            const int half_t  = t / 2;
            const int outer_r = r + half_t;
            const int inner_r = imax(0, r - half_t);

            const int min_x = cx - outer_r;
            const int max_x = cx + outer_r;
            const int min_y = cy - outer_r;
            const int max_y = cy + outer_r;

            const int r2       = r * r;
            const int inner_r2 = inner_r * inner_r;
            const int outer_r2 = outer_r * outer_r;

            const uchar3 fill =
                make_uchar3(circle.fill.x, circle.fill.y, circle.fill.z);
            const uchar3 border =
                make_uchar3(circle.border.x, circle.border.y, circle.border.z);

            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    const int dx    = x - cx;
                    const int dy    = y - cy;
                    const int dist2 = dx * dx + dy * dy;

                    if (t == 0) {
                        if (dist2 <= r2) {
                            set_pixel(img, img_size, x, y, fill);
                        }
                        continue;
                    }

                    if (circle.fill.w > 0 && dist2 <= inner_r2) {
                        set_pixel(img, img_size, x, y, fill);
                        continue;
                    }

                    if (circle.border.w > 0 && dist2 >= inner_r2 &&
                        dist2 <= outer_r2) {
                        set_pixel(img, img_size, x, y, border);
                    }
                }
            }
        }

        __global__ void build_glyph_instances_kernel(const Text*    texts,
                                                     int            count,
                                                     Font           font,
                                                     GlyphInstance* instances,
                                                     int maxTextLen) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= count) {
                return;
            }

            const Text&    txt = texts[idx];
            GlyphInstance* o   = instances + idx * maxTextLen;

            if (font.d_atlas_ptr == nullptr || font.d_glyphs_ptr == nullptr) {
                for (int i = 0; i < maxTextLen; ++i) {
                    o[i].glyphIndex = -1;
                }
                return;
            }

            int penX = static_cast<int>(txt.pos.x);
            int baseline =
                static_cast<int>(txt.pos.y) + static_cast<int>(font.ascent);

            int i = 0;
            for (; i < maxTextLen; ++i) {
                const char c = txt.msg[i];
                if (c == '\0') {
                    break;
                }

                const unsigned char uc = static_cast<unsigned char>(c);
                if (uc < kFirstChar || uc > kLastChar) {
                    const GlyphInfo space =
                        font.d_glyphs_ptr[' ' - kFirstChar];
                    penX += space.advance;
                    o[i].glyphIndex = -1;
                    continue;
                }

                const int        gidx = uc - kFirstChar;
                const GlyphInfo& g    = font.d_glyphs_ptr[gidx];

                o[i].glyphIndex = gidx;
                o[i].dstX       = penX + g.bearingX;
                o[i].dstY       = baseline - g.bearingY;
                o[i].color      = txt.color;

                penX += g.advance;
            }

            for (; i < maxTextLen; ++i) {
                o[i].glyphIndex = -1;
            }
        }

        __global__ void draw_glyph_instances_kernel(
            uchar*               img,
            uint2                shape,
            Font                 font,
            const GlyphInstance* instances,
            int                  totalInstances) {
            const int instanceIdx = blockIdx.x;
            if (instanceIdx >= totalInstances) {
                return;
            }

            const GlyphInstance inst = instances[instanceIdx];
            if (inst.glyphIndex < 0) {
                return;
            }

            const GlyphInfo g = font.d_glyphs_ptr[inst.glyphIndex];
            if (g.w <= 0 || g.h <= 0) {
                return;
            }

            for (int y = threadIdx.y; y < g.h; y += blockDim.y) {
                for (int x = threadIdx.x; x < g.w; x += blockDim.x) {
                    const int ix = inst.dstX + x;
                    const int iy = inst.dstY + y;

                    if (ix < 0 || ix >= shape.x || iy < 0 || iy >= shape.y) {
                        continue;
                    }

                    const int atlasIdx =
                        (g.y + y) * static_cast<int>(font.atlas_size.x) +
                        (g.x + x);
                    const uint a8 = font.d_atlas_ptr[atlasIdx];
                    if (a8 == 0) {
                        continue;
                    }

                    const float a = (static_cast<float>(a8) / 255.0f) *
                                    (static_cast<float>(inst.color.w) / 255.0f);
                    const int idx = iy * shape.x + ix;
                    const int rgb_idx = idx * 3;
                    uchar3 out = {img[rgb_idx + 0], img[rgb_idx + 1], img[rgb_idx + 2]};
                    out = blend_rgb(out, inst.color, a);
                    img[rgb_idx + 0] = out.x;
                    img[rgb_idx + 1] = out.y;
                    img[rgb_idx + 2] = out.z;
                }
            }
        }

        void drawText(Image3U&          img,
                      const Text::List& objs,
                      const Font*       font,
                      cudaStream_t      stream) {
            check_cuda();

            if (img.ptr() == nullptr || objs.empty() || font == nullptr ||
                font->d_atlas_ptr == nullptr || font->d_glyphs_ptr == nullptr) {
                return;
            }

            const int count = objs.size();

            Text* d_texts = nullptr;
            check_cuda(cudaMalloc(&d_texts,
                                  sizeof(Text) * static_cast<size_t>(count)));
            check_cuda(
                cudaMemcpyAsync(d_texts,
                                objs.data(),
                                sizeof(Text) * static_cast<size_t>(count),
                                cudaMemcpyHostToDevice,
                                stream));

            const size_t totalInstances = count * kMaxTextLen;

            check_cuda();
            GlyphInstance* d_instances = nullptr;
            check_cuda(cudaMalloc(&d_instances,
                                  totalInstances * sizeof(GlyphInstance)));

            const int  threads   = 128;
            const int  blocks    = (count + threads - 1) / threads;
            const Font fontValue = *font;

            build_glyph_instances_kernel<<<blocks, threads, 0, stream>>>(
                d_texts,
                count,
                fontValue,
                d_instances,
                kMaxTextLen);
            check_cuda();

            dim3 block(16, 16);
            dim3 grid(totalInstances, 1, 1);

            draw_glyph_instances_kernel<<<grid, block, 0, stream>>>(
                img.ptr(),
                img.getSize(),
                fontValue,
                d_instances,
                totalInstances);

            check_cuda();
            check_cuda(cudaFreeAsync(d_instances, stream));
            check_cuda(cudaFreeAsync(d_texts, stream));
            check_cuda();
        }

        void drawLine(Image3U&          img,
                      const Line::List& objs,
                      cudaStream_t      stream) {
            check_cuda();

            if (img.ptr() == nullptr || objs.empty()) {
                return;
            }

            const int count = objs.size();

            Line* d_lines = nullptr;
            check_cuda(
                cudaMallocAsync(&d_lines,
                                sizeof(Line) * static_cast<size_t>(count),
                                stream));
            check_cuda(
                cudaMemcpyAsync(d_lines,
                                objs.data(),
                                sizeof(Line) * static_cast<size_t>(count),
                                cudaMemcpyHostToDevice,
                                stream));

            const int block = 256;
            const int grid  = (count + block - 1) / block;

            draw_lines_kernel<<<grid, block, 0, stream>>>(img.ptr(),
                                                          img.getSize(),
                                                          d_lines,
                                                          count);
            check_cuda();
            check_cuda(cudaFreeAsync(d_lines, stream));
            check_cuda();
        }

        void drawRect(Image3U&          img,
                      const Rect::List& objs,
                      cudaStream_t      stream) {
            check_cuda();
            if (img.ptr() == nullptr || objs.empty()) {
                return;
            }

            const int count = objs.size();

            Rect* d_rects = nullptr;
            check_cuda(
                cudaMallocAsync(&d_rects,
                                sizeof(Rect) * static_cast<size_t>(count),
                                stream));
            check_cuda(
                cudaMemcpyAsync(d_rects,
                                objs.data(),
                                sizeof(Rect) * static_cast<size_t>(count),
                                cudaMemcpyHostToDevice,
                                stream));

            dim3 block(16, 16);
            dim3 grid((img.getSize().x + block.x - 1) / block.x,
                      (img.getSize().y + block.y - 1) / block.y);

            draw_rects_kernel<<<grid, block, 0, stream>>>(img.ptr(),
                                                          img.getSize(),
                                                          d_rects,
                                                          count);
            check_cuda();
            check_cuda(cudaFreeAsync(d_rects, stream));
            check_cuda();
        }

        void drawCircle(Image3U&            img,
                        const Circle::List& objs,
                        cudaStream_t        stream) {
            check_cuda();

            if (img.ptr() == nullptr || objs.empty()) {
                return;
            }

            const int count = objs.size();

            Circle* d_circles = nullptr;
            check_cuda(
                cudaMallocAsync(&d_circles,
                                sizeof(Circle) * static_cast<size_t>(count),
                                stream));
            check_cuda(
                cudaMemcpyAsync(d_circles,
                                objs.data(),
                                sizeof(Circle) * static_cast<size_t>(count),
                                cudaMemcpyHostToDevice,
                                stream));

            const int block = 256;
            const int grid  = (count + block - 1) / block;

            draw_circles_kernel<<<grid, block, 0, stream>>>(img.ptr(),
                                                            img.getSize(),
                                                            d_circles,
                                                            count);
            check_cuda();
            check_cuda(cudaFreeAsync(d_circles, stream));
            check_cuda();
        }

        void drawText(Image3U&     img,
                      Text*        d_text,
                      uint         count,
                      const Font*  font,
                      cudaStream_t stream) {
            check_cuda();

            if (img.ptr() == nullptr || d_text == nullptr || count == 0 ||
                font == nullptr || font->d_atlas_ptr == nullptr ||
                font->d_glyphs_ptr == nullptr) {
                return;
            }

            const size_t totalInstances =
                static_cast<size_t>(count) * kMaxTextLen;

            GlyphInstance* d_instances = nullptr;
            check_cuda(cudaMallocAsync(&d_instances,
                                       totalInstances * sizeof(GlyphInstance),
                                       stream));

            const int threads = 128;
            const int blocks =
                (static_cast<int>(count) + threads - 1) / threads;
            const Font fontValue = *font;

            build_glyph_instances_kernel<<<blocks, threads, 0, stream>>>(
                d_text,
                static_cast<int>(count),
                fontValue,
                d_instances,
                kMaxTextLen);

            check_cuda();
            dim3 block(16, 16);
            dim3 grid(totalInstances, 1, 1);

            draw_glyph_instances_kernel<<<grid, block, 0, stream>>>(
                img.ptr(),
                img.getSize(),
                fontValue,
                d_instances,
                static_cast<int>(totalInstances));

            check_cuda();
            check_cuda(cudaFreeAsync(d_instances, stream));
            check_cuda();
        }

        void drawLine(Image3U&     img,
                      Line*        d_lines,
                      uint         count,
                      cudaStream_t stream) {
            check_cuda();

            if (img.ptr() == nullptr || d_lines == nullptr || count == 0) {
                return;
            }

            const int block = 256;
            const int grid  = (static_cast<int>(count) + block - 1) / block;

            draw_lines_kernel<<<grid, block, 0, stream>>>(img.ptr(),
                                                          img.getSize(),
                                                          d_lines,
                                                          count);

            check_cuda();
        }

        void drawRect(Image3U&     img,
                      const Rect*  d_rects,
                      uint         count,
                      cudaStream_t stream) {
            check_cuda();

            if (img.ptr() == nullptr || d_rects == nullptr || count == 0) {
                return;
            }

            dim3 block(16, 16);
            dim3 grid((img.getSize().x + block.x - 1) / block.x,
                      (img.getSize().y + block.y - 1) / block.y);

            draw_rects_kernel<<<grid, block, 0, stream>>>(img.ptr(),
                                                          img.getSize(),
                                                          d_rects,
                                                          count);

            check_cuda();
        }

        void drawCircle(Image3U&     img,
                        Circle*      d_circles,
                        uint         count,
                        cudaStream_t stream) {
            check_cuda();

            if (img.ptr() == nullptr || d_circles == nullptr || count == 0) {
                return;
            }

            const int block = 256;
            const int grid  = (static_cast<int>(count) + block - 1) / block;

            draw_circles_kernel<<<grid, block, 0, stream>>>(img.ptr(),
                                                            img.getSize(),
                                                            d_circles,
                                                            count);

            check_cuda();
        }

    }  // namespace cuda
}  // namespace corekit
