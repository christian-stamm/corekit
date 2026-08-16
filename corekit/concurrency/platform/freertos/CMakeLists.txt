cmake_minimum_required(VERSION 3.25)

if(COREKIT_BUILD_PLATFORM_FREERTOS)

    add_library(corekit-concurrency-platform-freertos INTERFACE)

    target_include_directories(corekit-concurrency-platform-freertos 
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-concurrency-platform-freertos 
        INTERFACE
            corekit::concurrency::common
            FreeRTOS-Kernel-Heap4
    )

    add_library(corekit::concurrency ALIAS corekit-concurrency-platform-freertos)

endif()