cmake_minimum_required(VERSION 3.25)

if(COREKIT_BUILD_MCAP)

    find_package(LZ4 REQUIRED)

    add_library(mcap SHARED
        src/mcap.cpp
    )

    target_include_directories(mcap PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )

    target_link_libraries(mcap PUBLIC libzstd LZ4::lz4)

endif()