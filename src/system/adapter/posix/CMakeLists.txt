cmake_minimum_required(VERSION 3.25)

if(COREKIT_PLATFORM STREQUAL "POSIX")

    add_library(corekit-system-posix 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mutex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/smphr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/thread.cpp
    )

    target_include_directories(corekit-system-posix
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-system-posix
        PUBLIC
            corekit::system::common
    )

    add_library(corekit::system ALIAS corekit-system-posix)

endif()