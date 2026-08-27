cmake_minimum_required(VERSION 3.25)

macro(corekit_bsp_bootstrap)

    set(POSIX_RTOS_PORTABLE_PATH "${COREKIT_ROOT}/thirdparty/freertos/posix")

    if(NOT EXISTS ${POSIX_RTOS_PORTABLE_PATH})
        message(FATAL_ERROR "POSIX_RTOS_PORTABLE_PATH='${POSIX_RTOS_PORTABLE_PATH}' points to a non-existent directory. Please set it to the path of the FreeRTOS portable layer for the target platform.")
    endif()

endmacro()

macro(corekit_bsp_launch)

    add_subdirectory(${POSIX_RTOS_PORTABLE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/freertos)

endmacro()