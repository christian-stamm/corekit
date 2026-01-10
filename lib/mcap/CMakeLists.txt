cmake_minimum_required(VERSION 3.19)

include(FetchContent)
# Fetch and build lz4 library
set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_LEGACY_LZ4C OFF CACHE BOOL "" FORCE)
set(LZ4_BUILD_STATIC ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  lz4
  GIT_REPOSITORY https://github.com/lz4/lz4.git
  GIT_TAG dev
  SOURCE_SUBDIR build/cmake
)
FetchContent_MakeAvailable(lz4)

# Fetch and build zstd library
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  zstd
  GIT_REPOSITORY https://github.com/facebook/zstd.git
  GIT_TAG dev
  SOURCE_SUBDIR build/cmake
)
FetchContent_MakeAvailable(zstd)


FetchContent_Declare(
  mcap
  GIT_REPOSITORY https://github.com/foxglove/mcap.git
  GIT_TAG main
)
FetchContent_MakeAvailable(mcap)

add_library(mcap SHARED
    src/mcap.cpp
)

target_include_directories(mcap PUBLIC 
    ${mcap_SOURCE_DIR}/cpp/mcap/include
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_link_libraries(mcap PUBLIC libzstd_static lz4_static)