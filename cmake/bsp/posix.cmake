cmake_minimum_required(VERSION 3.25)

macro(corekit_bsp_bootstrap)

    set(BUILD_POSIX_MODULES OFF)

    if(${COREKIT_TARGET_DEVICE} STREQUAL "Posix")
        set(BUILD_POSIX_MODULES ON)
    endif()

endmacro()