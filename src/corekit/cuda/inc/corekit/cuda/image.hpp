#pragma once
#include <cuda_runtime.h>
#include <opencv2/core/hal/interface.h>

#include <cmath>
#include <memory>

#include "corekit/cuda/core.hpp"
#include "corekit/utils/assert.hpp"

#ifndef __CUDACC__
#    include <opencv4/opencv2/core/mat.hpp>
#    include <opencv4/opencv2/imgproc.hpp>
#endif

#include <vector_functions.h>
#include <vector_types.h>

#include <iostream>
#include <stdexcept>

namespace cv {
    class Mat;
}
namespace corekit {
    namespace cuda {

        using Size = uint2;

        inline bool operator==(const uint2& lhs, const uint2& rhs) {
            return lhs.x == rhs.x && lhs.y == rhs.y;
        }

        class Image3F;
        class Image3U;
        class Image4U;

        template <typename T = uchar, uint channels = 3>
        struct Image : public NvMem<T> {
            using Ptr = std::shared_ptr<Image<T>>;

            friend Image3F;
            friend Image3U;
            friend Image4U;

            Image(uint2 size = make_uint2(0, 0))
                : Image(NvMem<T>(size.x * size.y * channels), size) {}

            Image(NvMem<T> mem, uint2 size) : NvMem<T>(mem), size(size) {
                corecheck(
                    size.x * size.y * sizeof(T) * channels <= mem.get_bytes(),
                    "Provided memory is too small for the specified "
                    "image size");
            }

            Image(const Image<T, channels>& img) = default;
            Image(Image<T, channels>&& img)      = default;

            Image<T, channels>& operator=(const Image<T, channels>& img) =
                default;
            Image<T, channels>& operator=(Image<T, channels>&& img) = default;

            Image<T, channels>  clone() const;
            Image<T, channels>& clone_into(Image<T, channels>& out) const;
            uint2               getSize() const;

           protected:
            uint2 size;
        };

        class Image3U : public Image<uchar, 3> {
           public:
            using Image<uchar, 3>::Image;

            Image3U(const Image<uchar, 3>& img) : Image<uchar, 3>(img) {}

            Image3U(const Image3U& img)            = default;
            Image3U(Image3U&& img)                 = default;
            Image3U& operator=(const Image3U& img) = default;
            Image3U& operator=(Image3U&& img)      = default;

            cv::Mat         toCvMat() const;
            static Image3U  fromCvMat(const cv::Mat& img);
            static Image3U& fromCvMat(Image3U& out, const cv::Mat& img);

            uint8_t* toNv16(uint8_t* d_yuvData, bool swapUV = false) const;
            static Image3U  fromNv16(const Size&    size,
                                     const uint8_t* d_yuvData,
                                     bool           swapUV = false);
            static Image3U& fromNv16(Image3U&       out,
                                     const uint8_t* d_yuvData,
                                     bool           swapUV = false);

            Image3U  resize(uint2 size) const;
            Image3U& resize_into(Image3U& out, uint2 size) const;

            Image3U  pad(uint2 size, uchar3 value) const;
            Image3U& pad_into(Image3U& out, uchar3 value) const;

            Image3U  colflip() const;
            Image3U& colflip_into(Image3U& out) const;

            Image3F  chnflip() const;
            Image3F& chnflip_into(Image3F& out) const;

            Image4U  toRGBA() const;
            Image4U& toRGBA_into(Image4U& out) const;
        };

        class Image4U : public Image<uchar, 4> {
           public:
            using Image<uchar, 4>::Image;

            Image4U(const Image<uchar, 4>& img) : Image<uchar, 4>(img) {}

            Image4U(const Image4U& img)            = default;
            Image4U(Image4U&& img)                 = default;
            Image4U& operator=(const Image4U& img) = default;
            Image4U& operator=(Image4U&& img)      = default;

            Image3U  toRGB() const;
            Image3U& toRGB_into(Image3U& out) const;
        };

        class Image3F : public Image<float, 3> {
           public:
            using Image<float, 3>::Image;

            Image3F(const Image<float, 3>& img) : Image<float, 3>(img) {}

            Image3F(const Image3F& img)            = default;
            Image3F(Image3F&& img)                 = default;
            Image3F& operator=(const Image3F& img) = default;
            Image3F& operator=(Image3F&& img)      = default;

            Image3U  chnflip() const;
            Image3U& chnflip_into(Image3U& out) const;
        };

    }  // namespace cuda
}  // namespace corekit