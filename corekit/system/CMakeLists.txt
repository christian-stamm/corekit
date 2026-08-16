cmake_minimum_required(VERSION 3.25)

add_library(corekit-system INTERFACE)

target_include_directories(corekit-system
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:include>
)

add_library(corekit::system ALIAS corekit-system)