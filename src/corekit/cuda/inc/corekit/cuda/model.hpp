#pragma once

#include <NvInfer.h>
#include <vector_types.h>

#include <mutex>
#include <vector>

#include "corekit/types.hpp"
#include "corekit/utils/assert.hpp"
#include "corekit/utils/device.hpp"
#include "corekit/utils/filemgr.hpp"
#include "corekit/utils/logger.hpp"

namespace corekit {
    namespace cuda {

        using namespace corekit::types;
        using namespace corekit::utils;
        using namespace corekit::system;

        struct IOBinding {
            using List = std::vector<IOBinding>;

            Name           name  = Name();
            size_t         size  = 1;
            size_t         dsize = 1;
            nvinfer1::Dims dims;
            void*          ptr;
        };

        struct NvLogger : public nvinfer1::ILogger {
            NvLogger(const Name& name, Severity severity = Severity::kWARNING)
                : logger(name)
                , level(severity) {}

            void setLevel(Severity severity) noexcept {
                this->level = severity;
            }

            void log(Severity severity, const char* msg) noexcept override {
                if (this->level < severity) {
                    return;
                }

                switch (severity) {
                    case Severity::kINTERNAL_ERROR:
                        logger.fatal() << msg;
                        break;
                    case Severity::kERROR: logger.error() << msg; break;
                    case Severity::kWARNING: logger.warn() << msg; break;
                    case Severity::kINFO: logger.info() << msg; break;
                    default: logger.debug() << msg; break;
                }
            }

            Severity               level;
            corekit::utils::Logger logger;
        };

        struct Model : public Device {
            Model(const Path& engine)
                : Device(engine.stem())
                , nvlog(engine.stem())
                , file(engine) {}

            ~Model() {
                unload();
            }

            void info() const;

            cudaStream_t stream = 0;

           protected:
            virtual bool prepare() override;

            virtual bool cleanup() override;

            void exec() const;

            static int get_size_by_dims(const nvinfer1::Dims& dims);

            static int type_to_size(const nvinfer1::DataType& dataType);

            IOBinding::List inputs;
            IOBinding::List outputs;

            Path     file;
            NvLogger nvlog;

            nvinfer1::IRuntime*          runtime = nullptr;
            nvinfer1::ICudaEngine*       engine  = nullptr;
            nvinfer1::IExecutionContext* context = nullptr;
        };

    }  // namespace cuda
}  // namespace corekit
