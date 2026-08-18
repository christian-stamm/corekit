cmake_minimum_required(VERSION 3.25)

if(COREKIT_BUILD_PLATFORM_LINUX)

    add_library(corekit-concurrency-platform-linux INTERFACE)

    target_include_directories(corekit-concurrency-platform-linux
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-concurrency-platform-linux
        INTERFACE
            corekit::concurrency::common
    )

    add_library(corekit::concurrency ALIAS corekit-concurrency-platform-linux)

endif()