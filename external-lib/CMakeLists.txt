# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/fastcdr)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/glad)

if(COREKIT_BUILD_GTEST STREQUAL "ON")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gtest)
endif()

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pico-ocd)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mcap)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lz4)


# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pio-asm)



# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/zstd)