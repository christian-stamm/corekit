cmake_minimum_required(VERSION 3.25)

if(BUILD_COREKIT_CUDA)

    find_package(OpenCV REQUIRED)
    find_package(Freetype REQUIRED)

    set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT root directory")
    find_library(NVINFER_LIBRARY NAMES nvinfer PATHS ${TENSORRT_ROOT}/lib)
    find_library(NVONNXPARSER_LIBRARY NAMES nvonnxparser PATHS ${TENSORRT_ROOT}/lib)
    find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin PATHS ${TENSORRT_ROOT}/lib)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS ${TENSORRT_ROOT}/include)

    if(NOT NVINFER_LIBRARY OR NOT TENSORRT_INCLUDE_DIR)
        message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT to the TensorRT installation directory.")
    endif()

    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

    add_library(corekit-cuda STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/image.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/draw.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gltex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/model.cpp
    )

    add_dependencies(corekit-cuda
        corekit-core
        corekit-opengl
    )

    target_include_directories(corekit-cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_BINARY_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        /usr/local/cuda/include
    )

    target_link_libraries(corekit-cuda PUBLIC
        ${FREETYPE_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${NVINFER_LIBRARY}
        ${NVONNXPARSER_LIBRARY}
        ${NVINFER_PLUGIN_LIBRARY}
        corekit-core
        corekit-opengl
    )

endif()