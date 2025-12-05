cmake_minimum_required(VERSION 3.19)

include(FetchContent)

FetchContent_Declare(
  mcap
  URL https://github.com/foxglove/mcap/archive/refs/tags/releases/cpp/v2.0.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(mcap)

add_library(mcap 
    src/mcap.cpp
)

target_include_directories(mcap PUBLIC 
    ${mcap_SOURCE_DIR}/cpp/mcap/include
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_link_libraries(mcap PUBLIC zstd lz4)