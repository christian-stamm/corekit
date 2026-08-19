cmake_minimum_required(VERSION 3.25)

if(COREKIT_PAL_STDLIB)

    add_library(corekit-concurrency-stdlib INTERFACE)

    target_include_directories(corekit-concurrency-stdlib
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    add_library(corekit::concurrency::pal ALIAS corekit-concurrency-stdlib)

endif()