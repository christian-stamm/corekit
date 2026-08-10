cmake_minimum_required(VERSION 3.25)

add_library(corekit-core INTERFACE)

target_include_directories(corekit-core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(corekit-core
    INTERFACE
    corekit::platform
)

add_library(corekit::core ALIAS corekit-core)