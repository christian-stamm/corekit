cmake_minimum_required(VERSION 3.25)

if(${BUILD_FREERTOS_MODULES})
    add_subdirectory(core)
    add_subdirectory(pico)
    add_subdirectory(posix)
endif()