cmake_minimum_required(VERSION 3.25)

if(COREKIT_BUILD_PICO AND "${COREKIT_BASE_OS}" STREQUAL "BAREMETAL")

    add_library(corekit-concurrency-platform-pico 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mutex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/smphr.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/thread.cpp
    )

    target_include_directories(corekit-concurrency-platform-pico 
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(corekit-concurrency-platform-pico 
        PUBLIC
            corekit::concurrency::common
    )

    add_library(corekit::concurrency ALIAS corekit-concurrency-platform-pico)

endif()