/******************************************************************************
 *  Project: Trinity
 *
 *  Copyright (c) 2026 ZF Friedrichshafen AG. All rights reserved.
 *
 *  This software and associated documentation files ("Software") are the
 *  proprietary and confidential property of ZF Friedrichshafen AG. Unauthorized
 *  copying, distribution, modification, public display, or use of this
 *  Software, via any medium, is strictly prohibited without the prior
 *  written permission of ZF Friedrichshafen AG.
 *
 *  This Software is provided for internal use only and remains the exclusive
 *  intellectual property of ZF Friedrichshafen AG.
 *
 *  Author: Christian Stamm <christian.stamm@zf.com>
 *  Created: 2025-10
 *
 ******************************************************************************/

#include "corekit/cuda/gltex.hpp"

#include "corekit/cuda/image.hpp"
#include "corekit/utils/math.hpp"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err = (x);                                                 \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " ("     \
                      << static_cast<int>(err) << ") at " << __FILE__ << ":"   \
                      << __LINE__ << "\n";                                     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace corekit {
    namespace cuda {

        bool CudaTex::prepare() {
            if (!Texture::prepare())
                return false;

            // Register GL texture with CUDA
            CUDA_CHECK(cudaGraphicsGLRegisterImage(
                &cudaTexRes,
                tex,
                type,
                cudaGraphicsRegisterFlagsWriteDiscard));

            return true;
        }

        bool CudaTex::cleanup() {
            if (cudaTexRes != nullptr) {
                // Unregister GL texture with CUDA
                CUDA_CHECK(cudaGraphicsUnregisterResource(cudaTexRes));
                cudaTexRes = nullptr;
            }

            return Texture::cleanup();
        }

        void CudaTex::resize(Vec2 size, bool force) {
            // Check if we actually need to resize
            if (this->size == size && !force) {
                return;
            }

            // If CUDA resource is registered, we need to unregister it before
            // resizing the GL texture
            bool wasRegistered = (cudaTexRes != nullptr);

            if (wasRegistered) {
                CUDA_CHECK(cudaGraphicsUnregisterResource(cudaTexRes));
                cudaTexRes = nullptr;
            }

            // Call base class resize to reallocate the GL texture
            Texture::resize(size, force);

            // Re-register the texture with CUDA if it was previously registered
            if (wasRegistered) {
                CUDA_CHECK(cudaGraphicsGLRegisterImage(
                    &cudaTexRes,
                    tex,
                    type,
                    cudaGraphicsRegisterFlagsWriteDiscard));
            }
        }

        void CudaTex::fill(cv::Mat image, uint layer, FillMode mode) {
            if (image.empty())
                throw std::runtime_error(
                    "CudaTex::fill => empty image provided");

            switch (mode) {
                case FillMode::RESIZE_TEXTURE:
                    this->resize(Vec2(image.cols, image.rows));
                    break;
                case FillMode::RESIZE_IMAGE:
                    if (image.cols != size.x() || image.rows != size.y()) {
                        cv::resize(image, image, cv::Size(size.x(), size.y()));
                    }
                    break;
                default: break;
            }

            Image3U cudaimg(make_uint2(image.cols, image.rows));
            cudaimg.fromCvMat(image);
            CudaTex::fill(cudaimg.toRGBA(), layer);
        }

        void CudaTex::fill(Image4U image, uint layer, cudaStream_t stream) {
            if (depth < layer) {
                throw std::runtime_error(
                    "CudaTex::fill => layer index out of bounds");
            }

            const uint2 imdim = image.getSize();

            if ((int)(imdim.x) != (int)(size.x()) ||
                (int)(imdim.y) != (int)(size.y())) {
                throw std::runtime_error(
                    "CudaTex::fill => image dimensions do not match texture "
                    "size");
            }

            bool isLayered =
                (type == GL_TEXTURE_2D_ARRAY || type == GL_TEXTURE_3D);

            if (!isLayered && layer != 0) {
                throw std::runtime_error(
                    "CudaTex::fill => layer must be 0 for GL_TEXTURE_2D");
            }

            // Map GL texture as CUDA array
            CUDA_CHECK(cudaGraphicsMapResources(1, &cudaTexRes, stream));

            cudaArray_t cuArray     = nullptr;
            uint        targetLayer = isLayered ? layer : 0;
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cuArray,
                                                             cudaTexRes,
                                                             targetLayer,
                                                             0));

            // Copy from host to CUDA array (simple 2D copy)
            size_t rowBytes = imdim.x * 4 * sizeof(uchar);
            size_t height   = imdim.y;

            CUDA_CHECK(
                cudaMemcpy2DToArray(cuArray,  // dst array (GL texture storage)
                                    0,
                                    0,           // offsets
                                    image.ptr(), // src pointer (ext memory)
                                    rowBytes,    // src pitch in bytes
                                    rowBytes,    // width in bytes
                                    height,      // rows
                                    cudaMemcpyDeviceToDevice));  // direction

            CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaTexRes, stream));
        }

    };  // namespace cuda
};      // namespace corekit
