cmake_minimum_required(VERSION 3.25)

########################################################################

if(BUILD_COREKIT_CORE)

    find_package(Eigen3 REQUIRED)
    find_package(OpenCV REQUIRED)

    add_library(corekit-core STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/system/mutex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/system/smphr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/system/thread.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/assert.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/filemgr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/math.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/memory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/stream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/watch.cpp
    )

    target_include_directories(corekit-core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(corekit-core PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
    )

    if(COREKIT_PLATFORM STREQUAL "UNIX")
        target_compile_definitions(corekit-core PUBLIC PLATFORM_UNIX)
    elseif(COREKIT_PLATFORM STREQUAL "PICO")
        target_compile_definitions(corekit-core PUBLIC PLATFORM_PICO)
    endif()

endif()