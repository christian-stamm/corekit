cmake_minimum_required(VERSION 3.25)

add_library(corekit-iface INTERFACE)

target_include_directories(corekit-iface
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:include>
)

add_library(corekit::iface ALIAS corekit-iface)