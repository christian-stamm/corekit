cmake_minimum_required(VERSION 3.25)

if(COREKIT_PAL_BAREMETAL AND COREKIT_BSP_PICO)

    add_library(corekit-concurrency-pal-baremetal
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mutex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/smphr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/thread.cpp
    )

    target_include_directories(corekit-concurrency-pal-baremetal 
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-concurrency-pal-baremetal
        PUBLIC
            corekit::concurrency::api
            pico_stdlib
            pico_sync
    )

    add_library(corekit::concurrency::pal ALIAS corekit-concurrency-pal-baremetal)

endif()