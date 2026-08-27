cmake_minimum_required(VERSION 3.25)

if(${BUILD_PLATFORM_A_MODULES})
    add_subdirectory_aware(backendA)
endif()

if(${BUILD_PLATFORM_B_MODULES})
    add_subdirectory_aware(backendB)
endif()