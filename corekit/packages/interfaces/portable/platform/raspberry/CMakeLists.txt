cmake_minimum_required(VERSION 3.25)

if(${BUILD_PICO_MODULES})
    add_subdirectory_aware(pico)
endif()