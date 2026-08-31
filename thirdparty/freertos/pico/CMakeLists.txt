cmake_minimum_required(VERSION 3.25)

if(${BUILD_RP2XXX_MODULES})
    add_subdirectory(rp2040)
    add_subdirectory(rp2350)
endif()