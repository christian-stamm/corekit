#include <vector_functions.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/imgproc.hpp>
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

        __global__ void rgb_to_nv16_kernel(const uchar* in,
                                           uint8_t*     yPlane,
                                           uint8_t*     uvPlane,
                                           uint2        shape,
                                           bool         swapUV) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int idx     = y * shape.x + x;
            const int rgb_idx = idx * 3;

            const uint8_t r = in[rgb_idx + 0];
            const uint8_t g = in[rgb_idx + 1];
            const uint8_t b = in[rgb_idx + 2];

            const float rf = (float)(r);
            const float gf = (float)(g);
            const float bf = (float)(b);

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
                                           uchar*         out,
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

            const int idx    = y * shape.x + x;
            out[idx * 3 + 0] = clamp_u8(R);
            out[idx * 3 + 1] = clamp_u8(G);
            out[idx * 3 + 2] = clamp_u8(B);
        }

        __global__ void rgb_to_bgr_kernel(const uchar* in,
                                          uchar*       out,
                                          uint2        shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int idx     = y * shape.x + x;
            const int rgb_idx = idx * 3;

            out[rgb_idx + 0] = in[rgb_idx + 2];
            out[rgb_idx + 1] = in[rgb_idx + 1];
            out[rgb_idx + 2] = in[rgb_idx + 0];
        }

        __global__ void rgb_to_rgba_kernel(const uchar* in,
                                           uchar*       out,
                                           uint2        shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int idx      = y * shape.x + x;
            const int rgb_idx  = idx * 3;
            const int rgba_idx = idx * 4;

            out[rgba_idx + 0] = in[rgb_idx + 0];
            out[rgba_idx + 1] = in[rgb_idx + 1];
            out[rgba_idx + 2] = in[rgb_idx + 2];
            out[rgba_idx + 3] = 255;
        }

        __global__ void rgba_to_rgb_kernel(const uchar* in,
                                           uchar*       out,
                                           uint2        shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int idx      = y * shape.x + x;
            const int rgba_idx = idx * 4;
            const int rgb_idx  = idx * 3;

            out[rgb_idx + 0] = in[rgba_idx + 0];
            out[rgb_idx + 1] = in[rgba_idx + 1];
            out[rgb_idx + 2] = in[rgba_idx + 2];
        }

        __global__ void u3_to_f1_kernel(const uchar* in,
                                        float*       out,
                                        uint2        shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int idx     = y * shape.x + x;
            const int rgb_idx = idx * 3;
            const int plane   = shape.x * shape.y;

            const uint8_t r = in[rgb_idx + 0];
            const uint8_t g = in[rgb_idx + 1];
            const uint8_t b = in[rgb_idx + 2];

            out[0 * plane + idx] = (float)(r) / 255.0f;
            out[1 * plane + idx] = (float)(g) / 255.0f;
            out[2 * plane + idx] = (float)(b) / 255.0f;
        }

        __global__ void f1_to_u3_kernel(const float* in,
                                        uchar*       out,
                                        uint2        shape) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= shape.x || y >= shape.y) {
                return;
            }

            const int idx     = y * shape.x + x;
            const int rgb_idx = idx * 3;
            const int plane   = shape.x * shape.y;

            const float r = in[0 * plane + idx];
            const float g = in[1 * plane + idx];
            const float b = in[2 * plane + idx];

            out[rgb_idx + 0] = clamp_u8(static_cast<int>(lrintf(r * 255.0f)));
            out[rgb_idx + 1] = clamp_u8(static_cast<int>(lrintf(g * 255.0f)));
            out[rgb_idx + 2] = clamp_u8(static_cast<int>(lrintf(b * 255.0f)));
        }

        __global__ void resize_u3_kernel(const uchar* in,
                                         uchar*       out,
                                         uint2        inshape,
                                         uint2        outshape,
                                         float2       scale) {
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

            const int outIdx  = y * outshape.x + x;
            const int inIdx00 = y0 * inshape.x + x0;
            const int inIdx10 = y0 * inshape.x + x1;
            const int inIdx01 = y1 * inshape.x + x0;
            const int inIdx11 = y1 * inshape.x + x1;

            // Process each channel
            for (int c = 0; c < 3; ++c) {
                const float v00 = (float)in[inIdx00 * 3 + c];
                const float v10 = (float)in[inIdx10 * 3 + c];
                const float v01 = (float)in[inIdx01 * 3 + c];
                const float v11 = (float)in[inIdx11 * 3 + c];

                const float v0 = v00 + (v10 - v00) * dx;
                const float v1 = v01 + (v11 - v01) * dx;
                const float v  = v0 + (v1 - v0) * dy;

                out[outIdx * 3 + c] = clamp_u8(static_cast<int>(lrintf(v)));
            }
        }

        __global__ void pad_u3_kernel(const uchar* in,
                                      uchar*       out,
                                      uint2        inshape,
                                      uint2        outshape,
                                      uint2        padding,
                                      uchar3       padValue) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= outshape.x || y >= outshape.y) {
                return;
            }

            const int srcX = x - padding.x;
            const int srcY = y - padding.y;

            const int outIdx      = y * outshape.x + x;
            const int out_rgb_idx = outIdx * 3;

            if (srcX >= 0 && srcX < inshape.x && srcY >= 0 &&
                srcY < inshape.y) {
                const int srcIdx     = srcY * inshape.x + srcX;
                const int in_rgb_idx = srcIdx * 3;
                out[out_rgb_idx + 0] = in[in_rgb_idx + 0];
                out[out_rgb_idx + 1] = in[in_rgb_idx + 1];
                out[out_rgb_idx + 2] = in[in_rgb_idx + 2];
            } else {
                out[out_rgb_idx + 0] = padValue.x;
                out[out_rgb_idx + 1] = padValue.y;
                out[out_rgb_idx + 2] = padValue.z;
            }
        }

        /*
            #############################################################
        */

        template <typename T, uint channels>
        static void try_reuse(Image<T, channels>& out, uint2 size) {
            check_cuda();
            if (!out.ptr() || out.getSize() != size) {
                out = Image<T, channels>(size);
            }
            check_cuda();
        }

        Image3U Image3U::fromCvMat(const cv::Mat& img) {
            Image3U out;
            fromCvMat(out, img);
            return out;
        }

        Image3U& Image3U::fromCvMat(Image3U& out, const cv::Mat& img) {
            check_cuda();

            if (img.data == nullptr) {
                throw std::invalid_argument(
                    "Image::fromCvMat: input image data is null");
            }

            if (img.empty()) {
                throw std::invalid_argument(
                    "Image::fromCvMat: input image is empty");
            }

            if (img.type() != CV_8UC3) {
                throw std::invalid_argument(
                    "Image::fromCvMat: only CV_8UC3 type is supported");
            }

            if (img.isContinuous() == false) {
                throw std::invalid_argument(
                    "Image::fromCvMat: only continuous cv::Mat is supported");
            }

            if (out.ptr() == nullptr) {
                throw std::invalid_argument(
                    "Image::fromCvMat: output image is not initialized");
            }

            const Size size = make_uint2(img.cols, img.rows);

            try_reuse(out, size);

            check_cuda();
            check_cuda(cudaMemcpy(out.ptr(),
                                  img.data,
                                  out.get_bytes(),
                                  cudaMemcpyHostToDevice));

            check_cuda();
            return out;
        }

        cv::Mat Image3U::toCvMat() const {
            check_cuda();

            if (this->ptr() == nullptr || this->get_elems() == 0) {
                throw std::invalid_argument("Image::toCvMat: empty image");
            }

            cv::Mat img(size.y, size.x, CV_8UC3);
            check_cuda(cudaMemcpy(img.data,
                                  this->ptr(),
                                  this->get_bytes(),
                                  cudaMemcpyDeviceToHost));

            check_cuda();
            return img;
        }

        Image3U Image3U::fromNv16(const Size&    size,
                                  const uint8_t* d_yuvData,
                                  bool           swapUV) {
            Image3U out(size);
            fromNv16(out, d_yuvData, swapUV);
            return out;
        }

        Image3U& Image3U::fromNv16(Image3U&       out,
                                   const uint8_t* d_yuvData,
                                   bool           swapUV) {
            check_cuda();

            if (d_yuvData == nullptr) {
                throw std::invalid_argument(
                    "Image3U::fromNv16: input data is null");
            }

            if (out.size.x == 0 || out.size.y == 0) {
                throw std::invalid_argument(
                    "Image3U::fromNv16: output image size is zero");
            }

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(out.size, block);

            check_cuda();
            nv16_to_rgb_kernel<<<grid, block, 0>>>(
                d_yuvData,
                d_yuvData + out.size.x * out.size.y,
                out.ptr(),
                out.size,
                swapUV);

            check_cuda();
            return out;
        }

        uint8_t* Image3U::toNv16(uint8_t* d_yuvData, bool swapUV) const {
            check_cuda();

            if (!this->ptr()) {
                throw std::invalid_argument(
                    "Image3U::toNv16: input image data is null");
            }

            if (d_yuvData == nullptr) {
                throw std::invalid_argument(
                    "Image3U::toNv16: target buffer is null");
            }

            if (is_device_pointer(d_yuvData) == false) {
                throw std::invalid_argument(
                    "Image3U::toNv16: target buffer must be a device pointer");
            }

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            check_cuda();
            rgb_to_nv16_kernel<<<grid, block, 0>>>(this->ptr(),
                                                   d_yuvData,
                                                   d_yuvData + size.x * size.y,
                                                   size,
                                                   swapUV);

            check_cuda();

            return d_yuvData;
        }

        Image3U Image3U::resize(uint2 size) const {
            Image3U out(size);
            resize_into(out, size);
            return std::move(out);
        }

        Image3U& Image3U::resize_into(Image3U& out, uint2 size) const {
            check_cuda();

            if (!this->ptr()) {
                throw std::invalid_argument(
                    "Image3U::resize_into: input image data is null");
            }

            if (out.ptr() == this->ptr()) {
                throw std::invalid_argument(
                    "Image3U::resize_into: in-place resize is not supported");
            }

            try_reuse(out, size);

            const uint2  inshape  = this->size;
            const uint2  outshape = size;
            const float2 scale =
                make_float2((float)(inshape.x) / (float)(outshape.x),
                            (float)(inshape.y) / (float)(outshape.y));

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            check_cuda();
            resize_u3_kernel<<<grid, block>>>(this->ptr(),
                                              out.ptr(),
                                              inshape,
                                              outshape,
                                              scale);
            check_cuda();
            return out;
        }

        Image3U Image3U::pad(uint2 size, uchar3 value) const {
            Image3U out(size);
            pad_into(out, value);
            return std::move(out);
        }

        Image3U& Image3U::pad_into(Image3U& out, uchar3 value) const {
            check_cuda();

            if (!this->ptr()) {
                throw std::invalid_argument(
                    "Image3U::pad_into: input image data is null");
            }

            if (out.ptr() == this->ptr()) {
                throw std::invalid_argument(
                    "Image3U::pad_into: in-place pad is not supported");
            }

            const uint2 inshape  = this->size;
            const uint2 outshape = out.size;

            if (outshape.x < inshape.x || outshape.y < inshape.y) {
                throw std::invalid_argument(
                    "Image3U::pad_into: output size is smaller than input");
            }

            try_reuse(out, outshape);

            const uint2 padding = make_uint2((outshape.x - inshape.x) / 2,
                                             (outshape.y - inshape.y) / 2);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(outshape, block);

            check_cuda();
            pad_u3_kernel<<<grid, block>>>(this->ptr(),
                                           out.ptr(),
                                           inshape,
                                           outshape,
                                           padding,
                                           value);

            check_cuda();
            return out;
        }

        Image3U Image3U::colflip() const {
            Image3U out(size);
            colflip_into(out);
            return std::move(out);
        }

        Image3U& Image3U::colflip_into(Image3U& out) const {
            check_cuda();

            if (!this->ptr()) {
                throw std::invalid_argument(
                    "Image3U::colflip_into: input image data is null");
            }

            if (out.ptr() == this->ptr()) {
                throw std::invalid_argument(
                    "Image3U::colflip_into: in-place colflip is not supported");
            }

            try_reuse(out, size);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            check_cuda();
            rgb_to_bgr_kernel<<<grid, block>>>(this->ptr(), out.ptr(), size);
            check_cuda();

            return out;
        }

        Image4U Image3U::toRGBA() const {
            Image4U out(size);
            toRGBA_into(out);
            return std::move(out);
        }

        Image4U& Image3U::toRGBA_into(Image4U& out) const {
            check_cuda();

            if (!this->ptr()) {
                throw std::invalid_argument(
                    "Image3U::toRGBA_into: input image data is null");
            }

            try_reuse(out, size);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            check_cuda();
            rgb_to_rgba_kernel<<<grid, block>>>(this->ptr(), out.ptr(), size);
            check_cuda();

            return out;
        }

        Image3F Image3U::chnflip() const {
            Image3F out(size);
            chnflip_into(out);
            return std::move(out);
        }

        Image3F& Image3U::chnflip_into(Image3F& out) const {
            check_cuda();

            if (!this->ptr()) {
                throw std::invalid_argument(
                    "Image3U::chnflip_into: input image data is null");
            }

            try_reuse(out, size);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            check_cuda();
            u3_to_f1_kernel<<<grid, block>>>(this->ptr(), out.ptr(), size);
            check_cuda();

            return out;
        }

        Image3U Image3F::chnflip() const {
            Image3U out(size);
            chnflip_into(out);
            return std::move(out);
        }

        Image3U& Image3F::chnflip_into(Image3U& out) const {
            check_cuda();

            if (!this->ptr()) {
                throw std::invalid_argument(
                    "Image3F::chnflip_into: input image data is null");
            }

            // Verify Image3F has 3 channels worth of data (planar format)
            const uint64_t expected_elems = size.x * size.y * 3;
            if (this->get_elems() != expected_elems) {
                throw std::invalid_argument(
                    "Image3F::chnflip_into: Image3F does not have 3 channels");
            }

            try_reuse(out, size);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            check_cuda();
            f1_to_u3_kernel<<<grid, block>>>(this->ptr(), out.ptr(), size);
            check_cuda();

            return out;
        }

        Image3U Image4U::toRGB() const {
            Image3U out(size);
            toRGB_into(out);
            return std::move(out);
        }

        Image3U& Image4U::toRGB_into(Image3U& out) const {
            check_cuda();

            if (!this->ptr()) {
                throw std::invalid_argument(
                    "Image4U::toRGB_into: input image data is null");
            }

            try_reuse(out, size);

            const dim3 block = make_block_2d();
            const dim3 grid  = make_grid_2d(size, block);

            check_cuda();
            rgba_to_rgb_kernel<<<grid, block>>>(this->ptr(), out.ptr(), size);
            check_cuda();

            return out;
        }

        template <typename T, uint channels>
        Image<T, channels> Image<T, channels>::clone() const {
            Image<T, channels> out(size);
            this->clone_into(out);
            return std::move(out);
        }

        template <typename T, uint channels>
        Image<T, channels>& Image<T, channels>::clone_into(
            Image<T, channels>& out) const {
            try_reuse(out, size);

            check_cuda(cudaMemcpy(out.ptr(),
                                  this->ptr(),
                                  this->get_bytes(),
                                  cudaMemcpyDeviceToDevice));
            return out;
        }

        template <typename T, uint channels>
        uint2 Image<T, channels>::getSize() const {
            return size;
        }

        template struct Image<uchar, 3>;
        template struct Image<uchar, 4>;
        template struct Image<float, 3>;

    }  // namespace cuda
}  // namespace corekit
