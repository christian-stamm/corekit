cmake_minimum_required(VERSION 3.25)

if(COREKIT_PAL_FREERTOS)

    add_library(corekit-concurrency-freertos INTERFACE)

    target_include_directories(corekit-concurrency-freertos 
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-concurrency-freertos 
        INTERFACE
            FreeRTOS-Kernel-Heap4
    )

    add_library(corekit::concurrency::pal ALIAS corekit-concurrency-freertos)

endif()