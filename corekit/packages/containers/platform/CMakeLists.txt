cmake_minimum_required(VERSION 3.25)

if(${BUILD_STDLIB_MODULES})
    add_subdirectory(stdlib)
endif()

if(${BUILD_FREERTOS_MODULES})
    add_subdirectory(freertos)
endif()