cmake_minimum_required(VERSION 3.25)

if(COREKIT_PLATFORM STREQUAL "POSIX")

    add_library(corekit-system-posix INTERFACE)

    target_include_directories(corekit-system-posix
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-system-posix
        INTERFACE
            corekit::system::common
    )

    add_library(corekit::system ALIAS corekit-system-posix)

endif()