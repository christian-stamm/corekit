cmake_minimum_required(VERSION 3.19)

# build lz4 library
set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_STATIC ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/lz4/build/cmake)

# build zstd library
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/zstd/build/cmake)



add_library(mcap SHARED
    src/mcap.cpp
)

add_dependencies(mcap 
    lz4_static
    libzstd_static
)

target_include_directories(mcap PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_link_libraries(mcap PUBLIC libzstd_static lz4_static)