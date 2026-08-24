cmake_minimum_required(VERSION 3.25)

if(COREKIT_PAL_STDLIB)

    add_library(corekit-concurrency-pal-stdlib
        ${CMAKE_CURRENT_SOURCE_DIR}/src/smphr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/thread.cpp
    )

    target_include_directories(corekit-concurrency-pal-stdlib
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-concurrency-pal-stdlib
        PUBLIC
            corekit::concurrency::api
    )

    add_library(corekit::concurrency::pal ALIAS corekit-concurrency-pal-stdlib)

endif()