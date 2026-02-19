#include <vector_functions.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "corekit/cuda/core.hpp"
#include "corekit/cuda/image.hpp"

namespace corekit {
    namespace cuda {

        constexpr int kBlockSize2D = 16;

        __device__ __forceinline__ uint8_t clamp_u8(int     v,
                                                    uint8_t lo = 0,
                                                    uint8_t hi = 255) {
            return static_cast<uint8_t>(v < lo ? lo : (v > hi ? hi : v));
        }

        __device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
            return v < lo ? lo : (v > hi ? hi : v);
        }

        dim3 make_block_2d() {
            return dim3(kBlockSize2D, kBlockSize2D);
        }

        dim3 make_grid_2d(uint2 size, dim3 block) {
            return dim3((size.x + block.x - 1) / block.x,
                        (size.y + block.y - 1) / block.y);
        }

        template <typename T>
        static void ensure_image(Image<T>& out,
                                 uint2     size,
                                 Format    format,
                                 Layout    layout) {
            if (!out.data || out.getSize().x != size.x ||
                out.getSize().y != size.y || out.getFormat() != format ||
                out.getLayout() != layout) {
                out = Image<T>(nullptr, size, format, layout);
            }
        }

        __global__ void rgb_to_nv16_kernel(const uchar3* in,
                                           uint8_t*      yPlane,
                                           uint8_t*      uvPlane,
                                           uint2         shape,
                                           bool          swapUV) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int    idx = y * shape.x + x;
            const uchar3 rgb = in[idx];

            const float rf = (float)(rgb.x);
            const float gf = (float)(rgb.y);
            const float bf = (float)(rgb.z);

            const float yf = 0.299f * rf + 0.587f * gf + 0.114f * bf;
            const float uf =
                -0.168736f * rf - 0.331264f * gf + 0.5f * bf + 128.0f;
            const float vf =
                0.5f * rf - 0.418688f * gf - 0.081312f * bf + 128.0f;

            yPlane[idx] = clamp_u8(static_cast<int>(lrintf(yf)));

            const int pair = x & ~1;
            const int Uidx = swapUV ? 1 : 0;
            const int Vidx = swapUV ? 0 : 1;

            if ((x & 1) == 0) {
                uvPlane[y * shape.x + pair + Uidx] =
                    clamp_u8(static_cast<int>(lrintf(uf)));
                uvPlane[y * shape.x + pair + Vidx] =
                    clamp_u8(static_cast<int>(lrintf(vf)));
            }
        }

        __global__ void nv16_to_rgb_kernel(const uint8_t* yPlane,
                                           const uint8_t* uvPlane,
                                           uchar3*        out,
                                           uint2          shape,
                                           bool           swapUV) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const uint8_t* yrow  = yPlane + y * shape.x;
            const uint8_t* uvrow = uvPlane + y * shape.x;

            const int pair = x & ~1;
            const int Uidx = swapUV ? 1 : 0;
            const int Vidx = swapUV ? 0 : 1;

            const int U = uvrow[pair + Uidx];
            const int V = uvrow[pair + Vidx];
            const int Y = yrow[x];

            const float yf = (float)(Y);
            const float uf = (float)(U)-128.0f;
            const float vf = (float)(V)-128.0f;

            const float rf = yf + 1.402f * vf;
            const float gf = yf - 0.344136f * uf - 0.714136f * vf;
            const float bf = yf + 1.772f * uf;

            const int R = static_cast<int>(lrintf(rf));
            const int G = static_cast<int>(lrintf(gf));
            const int B = static_cast<int>(lrintf(bf));

            out[y * shape.x + x] =
                make_uchar3(clamp_u8(R), clamp_u8(G), clamp_u8(B));
        }

        __global__ void rgb_to_bgr_kernel(const uchar3* in,
                                          uchar3*       out,
                                          uint2         shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int    idx = y * shape.x + x;
            const uchar3 rgb = in[idx];
            out[idx]         = make_uchar3(rgb.z, rgb.y, rgb.x);
        }

        __global__ void rgb_to_rgba_kernel(const uchar3* in,
                                           uchar4*       out,
                                           uint2         shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int    idx = y * shape.x + x;
            const uchar3 rgb = in[idx];
            out[idx]         = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
        }

        __global__ void rgba_to_rgb_kernel(const uchar4* in,
                                           uchar3*       out,
                                           uint2         shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int    idx  = y * shape.x + x;
            const uchar4 rgba = in[idx];
            out[idx]          = make_uchar3(rgba.x, rgba.y, rgba.z);
        }

        __global__ void u3_to_f1_kernel(const uchar3* in,
                                        float*        out,
                                        uint2         shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int    idx   = y * shape.x + x;
            const uchar3 v     = in[idx];
            const int    plane = shape.x * shape.y;

            out[0 * plane + idx] = (float)(v.x) / 255.0f;
            out[1 * plane + idx] = (float)(v.y) / 255.0f;
            out[2 * plane + idx] = (float)(v.z) / 255.0f;
        }

        __global__ void f1_to_u3_kernel(const float* in,
                                        uchar3*      out,
                                        uint2        shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int idx   = y * shape.x + x;
            const int plane = shape.x * shape.y;

            const float r = in[0 * plane + idx];
            const float g = in[1 * plane + idx];
            const float b = in[2 * plane + idx];

            out[idx] =
                make_uchar3(clamp_u8(static_cast<int>(lrintf(r * 255.0f))),
                            clamp_u8(static_cast<int>(lrintf(g * 255.0f))),
                            clamp_u8(static_cast<int>(lrintf(b * 255.0f))));
        }

        __global__ void resize_u3_kernel(const uchar3* in,
                                         uchar3*       out,
                                         uint2         inshape,
                                         uint2         outshape,
                                         float2        scale) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= outshape.x || y >= outshape.y) {
                return;
            }

            const float srcX = ((float)(x) + 0.5f) * scale.x - 0.5f;
            const float srcY = ((float)(y) + 0.5f) * scale.y - 0.5f;

            const int x0 =
                clamp_int(static_cast<int>(floorf(srcX)), 0, inshape.x - 1);
            const int y0 =
                clamp_int(static_cast<int>(floorf(srcY)), 0, inshape.y - 1);
            const int x1 = clamp_int(x0 + 1, 0, inshape.x - 1);
            const int y1 = clamp_int(y0 + 1, 0, inshape.y - 1);

            const float dx = srcX - (float)(x0);
            const float dy = srcY - (float)(y0);

            const int outIdx  = (y * outshape.x + x);
            const int inIdx00 = (y0 * inshape.x + x0);
            const int inIdx10 = (y0 * inshape.x + x1);
            const int inIdx01 = (y1 * inshape.x + x0);
            const int inIdx11 = (y1 * inshape.x + x1);

            const uchar3 v00 = in[inIdx00];
            const uchar3 v10 = in[inIdx10];
            const uchar3 v01 = in[inIdx01];
            const uchar3 v11 = in[inIdx11];

            const float3 v0 = make_float3(
                (float)(v00.x) + ((float)(v10.x) - (float)(v00.x)) * dx,
                (float)(v00.y) + ((float)(v10.y) - (float)(v00.y)) * dx,
                (float)(v00.z) + ((float)(v10.z) - (float)(v00.z)) * dx);

            const float3 v1 = make_float3(
                (float)(v01.x) + ((float)(v11.x) - (float)(v01.x)) * dx,
                (float)(v01.y) + ((float)(v11.y) - (float)(v01.y)) * dx,
                (float)(v01.z) + ((float)(v11.z) - (float)(v01.z)) * dx);

            const float3 v = make_float3(v0.x + (v1.x - v0.x) * dy,
                                         v0.y + (v1.y - v0.y) * dy,
                                         v0.z + (v1.z - v0.z) * dy);

            out[outIdx] = make_uchar3(clamp_u8(static_cast<int>(lrintf(v.x))),
                                      clamp_u8(static_cast<int>(lrintf(v.y))),
                                      clamp_u8(static_cast<int>(lrintf(v.z))));
        }

        __global__ void pad_u3_kernel(const uchar3* in,
                                      uchar3*       out,
                                      uint2         inshape,
                                      uint2         outshape,
                                      uint2         padding,
                                      uchar3        padValue) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= outshape.x || y >= outshape.y) {
                return;
            }

            const int srcX = x - padding.x;
            const int srcY = y - padding.y;

            const int outIdx = y * outshape.x + x;

            if (srcX >= 0 && srcX < inshape.x && srcY >= 0 &&
                srcY < inshape.y) {
                out[outIdx] = in[srcY * inshape.x + srcX];
            } else {
                out[outIdx] = padValue;
            }
        }

        Image3U Image3U::fromNv16(const uint8_t* h_yuvData,
                                  uint2          size,
                                  bool           swapUV) {
            Image3U out(nullptr, size, RGB, HWC);
            fromNv16_into(h_yuvData, size, out, swapUV, nullptr);
            return out;
        }

        void Image3U::fromNv16_into(const uint8_t* h_yuvData,
                                    uint2          size,
                                    Image3U&       out,
                                    bool           swapUV,
                                    uint8_t*       dScratch) {
            check_cuda();

            if (h_yuvData == nullptr) {
                throw std::invalid_argument(
                    "Image3U::fromNv16_into: input data is null");
            }

            ensure_image(out, size, RGB, HWC);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            uint8_t* d_yPlane  = nullptr;
            uint8_t* d_uvPlane = nullptr;

            if (dScratch) {
                d_yPlane  = dScratch;
                d_uvPlane = dScratch + size.x * size.y;
            } else {
                check_cuda(cudaMalloc(&d_yPlane, size.x * size.y));
                check_cuda(cudaMalloc(&d_uvPlane, size.x * size.y));
            }

            check_cuda(cudaMemcpy(d_yPlane,
                                  h_yuvData,
                                  size.x * size.y,
                                  cudaMemcpyHostToDevice));

            check_cuda(cudaMemcpy(d_uvPlane,
                                  h_yuvData + size.x * size.y,
                                  size.x * size.y,
                                  cudaMemcpyHostToDevice));

            nv16_to_rgb_kernel<<<grid, block, 0>>>(d_yPlane,
                                                   d_uvPlane,
                                                   out.data,
                                                   size,
                                                   swapUV);

            if (!dScratch) {
                cudaFree(d_yPlane);
                cudaFree(d_uvPlane);
            }

            check_cuda();
        }

        uint8_t* Image3U::toNv16(uint8_t* target, bool swapUV) const {
            if (target == nullptr) {
                target = new uint8_t[size.x * size.y * 2];
            }

            toNv16_into(target, swapUV, nullptr);
            return target;
        }

        void Image3U::toNv16_into(uint8_t* target,
                                  bool     swapUV,
                                  uint8_t* dScratch) const {
            check_cuda();

            if (!data) {
                throw std::invalid_argument(
                    "Image3U::toNv16_into: input image data is null");
            }

            if (target == nullptr) {
                throw std::invalid_argument(
                    "Image3U::toNv16_into: target pointer is null");
            }

            uint8_t* d_yuvData = nullptr;

            if (dScratch) {
                d_yuvData = dScratch;
            } else {
                check_cuda(cudaMalloc(&d_yuvData, size.x * size.y * 2));
            }

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            rgb_to_nv16_kernel<<<grid, block, 0>>>(data,
                                                   d_yuvData,
                                                   d_yuvData + size.x * size.y,
                                                   size,
                                                   swapUV);

            check_cuda(cudaMemcpy(target,
                                  d_yuvData,
                                  size.x * size.y * 2,
                                  cudaMemcpyDeviceToHost));

            if (!dScratch) {
                cudaFree(d_yuvData);
            }

            check_cuda();
        }

        Image3U Image3U::resize(uint2 size) const {
            Image3U out(nullptr, size, getFormat(), getLayout());
            resize_into(out, size);
            return out;
        }

        void Image3U::resize_into(Image3U& out, uint2 size) const {
            if (!data) {
                throw std::invalid_argument(
                    "Image3U::resize_into: input image data is null");
            }

            if (getLayout() != HWC) {
                throw std::invalid_argument(
                    "Image3U::resize_into: only HWC layout is supported");
            }

            if (out.data == data) {
                throw std::invalid_argument(
                    "Image3U::resize_into: in-place resize is not supported");
            }

            ensure_image(out, size, getFormat(), getLayout());

            const uint2  shape = getSize();
            const float2 scale =
                make_float2((float)(shape.x) / (float)(size.x),
                            (float)(shape.y) / (float)(size.y));

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            resize_u3_kernel<<<grid, block>>>(data,
                                              out.data,
                                              shape,
                                              size,
                                              scale);
        }

        Image3U Image3U::pad(uint2 size, uchar3 value) const {
            Image3U out(nullptr, size, getFormat(), getLayout());
            pad_into(out, size, value);
            return out;
        }

        void Image3U::pad_into(Image3U& out, uint2 size, uchar3 value) const {
            check_cuda();

            if (!data) {
                throw std::invalid_argument(
                    "Image3U::pad_into: input image data is null");
            }

            if (getLayout() != HWC) {
                throw std::invalid_argument(
                    "Image3U::pad_into: only HWC layout is supported");
            }

            if (out.data == data) {
                throw std::invalid_argument(
                    "Image3U::pad_into: in-place pad is not supported");
            }

            ensure_image(out, size, getFormat(), getLayout());

            const uint2 inshape = getSize();
            const uint2 padding =
                make_uint2((size.x - inshape.x) / 2, (size.y - inshape.y) / 2);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            pad_u3_kernel<<<grid, block>>>(data,
                                           out.data,
                                           inshape,
                                           size,
                                           padding,
                                           value);

            check_cuda();
        }

        Image3U Image3U::colflip() const {
            Image3U out(nullptr, getSize(), getFormat(), getLayout());
            colflip_into(out);
            return out;
        }

        void Image3U::colflip_into(Image3U& out) const {
            check_cuda();

            if (!data) {
                throw std::invalid_argument(
                    "Image3U::colflip_into: input image data is null");
            }

            if (getLayout() != HWC) {
                throw std::invalid_argument(
                    "Image3U::colflip_into: only HWC layout is supported");
            }

            if (getFormat() != RGB && getFormat() != BGR) {
                throw std::invalid_argument(
                    "Image3U::colflip_into: only RGB/BGR formats are "
                    "supported");
            }

            const Format outFormat = (getFormat() == RGB) ? BGR : RGB;
            ensure_image(out, getSize(), outFormat, getLayout());

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(getSize(), block);

            rgb_to_bgr_kernel<<<grid, block>>>(data, out.data, getSize());
            check_cuda();
        }

        Image3U::Image4U Image3U::toRGBA() const {
            Image4U out(nullptr, getSize(), RGBA, HWC);
            toRGBA_into(out);
            return out;
        }

        void Image3U::toRGBA_into(Image4U& out) const {
            check_cuda();

            if (!data) {
                throw std::invalid_argument(
                    "Image3U::toRGBA_into: input image data is null");
            }

            if (getLayout() != HWC) {
                throw std::invalid_argument(
                    "Image3U::toRGBA_into: only HWC layout is supported");
            }

            if (getFormat() != RGB && getFormat() != BGR) {
                throw std::invalid_argument(
                    "Image3U::toRGBA_into: only RGB/BGR formats are supported");
            }

            ensure_image(out, getSize(), RGBA, HWC);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(getSize(), block);

            rgb_to_rgba_kernel<<<grid, block>>>(data, out.data, getSize());
            check_cuda();
        }

        Image3U::Image1F Image3U::chnflip() const {
            Image<float> out(nullptr, getSize(), getFormat(), CHW);
            chnflip_into(out);
            return out;
        }

        void Image3U::chnflip_into(Image1F& out) const {
            check_cuda();

            if (!data) {
                throw std::invalid_argument(
                    "Image3U::chnflip_into: input image data is null");
            }

            if (getLayout() != HWC) {
                throw std::invalid_argument(
                    "Image3U::chnflip_into: only HWC layout is supported");
            }

            if (getFormat() != RGB && getFormat() != BGR) {
                throw std::invalid_argument(
                    "Image3U::chnflip_into: only RGB/BGR formats are "
                    "supported");
            }

            if (getSize().x == 0 || getSize().y == 0) {
                throw std::invalid_argument(
                    "Image3U::chnflip_into: input image has zero width or "
                    "height");
            }

            ensure_image(out, getSize(), getFormat(), CHW);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(getSize(), block);

            u3_to_f1_kernel<<<grid, block>>>(data, out.data, getSize());
            check_cuda();
        }

        Image1F::Image3U Image1F::chnflip() const {
            Image<uchar3> out(nullptr, getSize(), getFormat(), HWC);
            chnflip_into(out);
            return out;
        }

        void Image1F::chnflip_into(Image3U& out) const {
            check_cuda();

            if (!data) {
                throw std::invalid_argument(
                    "Image1F::chnflip_into: input image data is null");
            }

            if (getLayout() != CHW) {
                throw std::invalid_argument(
                    "Image1F::chnflip_into: only CHW layout is supported");
            }

            if (getFormat() != RGB && getFormat() != BGR) {
                throw std::invalid_argument(
                    "Image1F::chnflip_into: only RGB/BGR formats are "
                    "supported");
            }

            ensure_image(out, getSize(), getFormat(), HWC);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(getSize(), block);

            f1_to_u3_kernel<<<grid, block>>>(data, out.data, getSize());
            check_cuda();
        }

        template struct Image<uchar3>;
        template struct Image<uchar4>;
        template struct Image<float>;

    }  // namespace cuda
}  // namespace corekit
