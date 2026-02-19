#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <memory>

#include "corekit/cuda/core.hpp"

#ifndef __CUDACC__
#    include <opencv4/opencv2/core/mat.hpp>
#    include <opencv4/opencv2/imgproc.hpp>
#endif

#include <vector_functions.h>
#include <vector_types.h>

#include <iostream>
#include <stdexcept>
namespace corekit {
    namespace cuda {

        enum Format {
            RGB,
            BGR,
            RGBA,
        };

        enum Layout {
            HWC,
            CHW,
        };

        template <typename T = uchar3>
        struct Image {
            using Ptr = std::shared_ptr<Image<T>>;

            friend class Image3U;
            friend class Image4U;
            friend class Image1F;

            Image(T*     data   = nullptr,
                  uint2  size   = make_uint2(0, 0),
                  Format format = RGB,
                  Layout layout = HWC)
                : Image(data,
                        size,
                        format,
                        layout,
                        (data == nullptr) ? new int(0) : nullptr) {}

            Image(const Image& other)
                : Image(other.data,
                        other.size,
                        other.format,
                        other.layout,
                        other.owners) {}

            Image(Image&& other) noexcept
                : data(other.data)
                , size(std::move(other.size))
                , pixels(std::move(other.pixels))
                , format(std::move(other.format))
                , layout(std::move(other.layout))
                , owners(other.owners) {
                other.data   = nullptr;
                other.owners = nullptr;
            }

            Image<T>& operator=(const Image& other) {
                if (this != &other) {
                    release();

                    data   = other.data;
                    size   = other.size;
                    pixels = other.pixels;
                    format = other.format;
                    layout = other.layout;
                    owners = other.owners;

                    request();
                }

                return *this;
            }

            Image<T>& operator=(Image&& other) {
                if (this != &other) {
                    this->release();

                    data   = other.data;
                    size   = std::move(other.size);
                    pixels = std::move(other.pixels);
                    format = std::move(other.format);
                    layout = std::move(other.layout);
                    owners = other.owners;

                    other.data   = nullptr;
                    other.owners = nullptr;
                }

                return *this;
            }

            ~Image() {
                release();
            }

            Image<T> clone() const {
                Image<T>     copy(nullptr, size, format, layout, owners);
                const size_t bytes = element_count() * sizeof(T);

                cudaMalloc(&copy.data, bytes);
                cudaMemcpy(copy.data,
                           this->data,
                           bytes,
                           cudaMemcpyDeviceToDevice);

                return copy;
            }

            uint2 getSize() const {
                return size;
            }

            Format getFormat() const {
                return format;
            }

            Layout getLayout() const {
                return layout;
            }

            size_t getPixels() const {
                return pixels;
            }

            T* data;

           protected:
            Image(T*     data,
                  uint2  size,
                  Format format,
                  Layout layout,
                  int*   owners)
                : data(data)
                , size(size)
                , pixels(size.x * size.y)
                , format(format)
                , layout(layout)
                , owners(owners) {
                if (!owners && !cuda::is_device_pointer(data)) {
                    throw std::invalid_argument(
                        "Image constructor: data pointer must be a cuda "
                        "pointer");
                }

                request();
            }

            void request() {
                if (owners) {
                    if ((*owners == 0) && (0 < pixels)) {
                        cudaMalloc(&data, element_count() * sizeof(T));
                    }

                    (*owners) += 1;
                }
            }

            void release() {
                if (owners) {
                    (*owners) -= 1;

                    if ((*owners == 0) && data) {
                        cudaFree(data);
                        data = nullptr;
                    }
                }
            }

            size_t element_count() const {
                const size_t base = size.x * size.y;

                if constexpr (std::is_same_v<T, float>) {
                    const size_t channels = (format == RGBA) ? 4 : 3;
                    return base * channels;
                }

                return base;
            }

            uint2  size;
            size_t pixels;
            Format format;
            Layout layout;
            int*   owners;
        };

        class Image3U : public Image<uchar3> {
           public:
            using Image1F = Image<float>;
            using Image4U = Image<uchar4>;
            using Image<uchar3>::Image;

            Image3U(const Image<uchar3>& img)
                : Image(img.data,
                        img.size,
                        img.format,
                        img.layout,
                        img.owners) {}

            Image3U(const Image3U& img)
                : Image(img.data,
                        img.size,
                        img.format,
                        img.layout,
                        img.owners) {}

#ifndef __CUDACC__
            static Image3U fromCvMat(const cv::Mat& img, Format fmt = BGR) {
                if (img.empty()) {
                    throw std::invalid_argument(
                        "Image::fromCvMat: input image is empty");
                }

                if (img.type() != CV_8UC3) {
                    throw std::invalid_argument(
                        "Image::fromCvMat: only CV_8UC3 type is supported");
                }

                Image3U out(nullptr, make_uint2(img.cols, img.rows), fmt, HWC);
                cudaMemcpy(out.data,
                           img.data,
                           img.cols * img.rows * sizeof(uchar3),
                           cudaMemcpyHostToDevice);

                return out;
            }

            cv::Mat toCvMat() const {
                if (data == nullptr || pixels == 0) {
                    std::cerr << "Image::toCvMat: empty image" << std::endl;
                    return cv::Mat();
                }

                cv::Mat img(size.y, size.x, CV_8UC3);
                cudaMemcpy(img.data,
                           this->data,
                           pixels * sizeof(uchar3),
                           cudaMemcpyDeviceToHost);

                return img;
            }
#endif
            static Image3U fromNv16(const uint8_t* yuvData,
                                    uint2          size,
                                    bool           swapUV = false);

            static void fromNv16_into(const uint8_t* yuvData,
                                      uint2          size,
                                      Image3U&       out,
                                      bool           swapUV   = false,
                                      uint8_t*       dScratch = nullptr);

            uint8_t* toNv16(uint8_t* target = nullptr,
                            bool     swapUV = false) const;

            void toNv16_into(uint8_t* target,
                             bool     swapUV   = false,
                             uint8_t* dScratch = nullptr) const;

            Image3U resize(uint2 size) const;
            void    resize_into(Image3U& out, uint2 size) const;
            Image3U pad(uint2  size,
                        uchar3 value = make_uchar3(114, 114, 114)) const;
            void    pad_into(Image3U& out,
                             uint2    size,
                             uchar3   value = make_uchar3(114, 114, 114)) const;
            Image3U colflip() const;
            void    colflip_inplace();
            void    colflip_into(Image3U& out) const;
            Image1F chnflip() const;
            void    chnflip_into(Image1F& out) const;
            Image4U toRGBA() const;
            void    toRGBA_into(Image4U& out) const;
        };

        class Image4U : public Image<uchar4> {
           public:
            using Image3U = Image<uchar3>;
            using Image<uchar4>::Image;

            Image4U(const Image<uchar4>& img)
                : Image(img.data,
                        img.size,
                        img.format,
                        img.layout,
                        img.owners) {}

            Image4U toRGB() const;
        };

        class Image1F : public Image<float> {
           public:
            using Image3U = Image<uchar3>;
            using Image<float>::Image;

            Image1F(const Image<float>& img)
                : Image(img.data,
                        img.size,
                        img.format,
                        img.layout,
                        img.owners) {}

            Image1F(const Image1F& img)
                : Image(img.data,
                        img.size,
                        img.format,
                        img.layout,
                        img.owners) {}

            Image3U chnflip() const;
            void    chnflip_into(Image3U& out) const;
        };

    }  // namespace cuda
}  // namespace corekit