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

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/videoio.hpp>

#include "corekit/cuda/image.hpp"
#include "corekit/opengl/texture.hpp"

// --- CUDA ---
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace corekit {
    namespace cuda {

        using namespace corekit::types;
        using namespace corekit::utils;
        using namespace corekit::opengl;

        class CudaTex : public Texture {
           public:
            using Ptr  = std::shared_ptr<CudaTex>;
            using List = std::vector<Ptr>;
            using Map  = std::map<Hash, Ptr>;

            CudaTex(const Texture::Settings& settings) : Texture(settings) {}

            virtual void resize(Vec2 size, bool force = false) override;
            virtual void fill(cv::Mat  image,
                              uint     layer = 0,
                              FillMode mode  = RESIZE_IMAGE) override;

            void fill(Image4U image, uint layer = 0, cudaStream_t stream = 0);

           protected:
            virtual bool prepare() override;
            virtual bool cleanup() override;

           private:
            cudaGraphicsResource* cudaTexRes = nullptr;
        };

    };  // namespace cuda
};      // namespace corekit
