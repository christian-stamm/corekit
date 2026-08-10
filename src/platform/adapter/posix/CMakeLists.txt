cmake_minimum_required(VERSION 3.25)

if(COREKIT_PLATFORM STREQUAL "POSIX")

    add_library(corekit-iface-posix INTERFACE)

    target_include_directories(corekit-iface-posix
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-iface-posix
        INTERFACE
            corekit::iface
    )

    add_library(corekit::platform ALIAS corekit-iface-posix)

endif()