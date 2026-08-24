cmake_minimum_required(VERSION 3.25)

set(FREERTOS_KERNEL_PATH $ENV{FREERTOS_KERNEL_PATH} CACHE PATH "Path to the FreeRTOS kernel source code")
set(FREERTOS_PORTABLE_IMPL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/portable/${COREKIT_TARGET}" CACHE PATH "Path to the FreeRTOS portable implementation for the selected target")


add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/modules")
add_subdirectory("${FREERTOS_PORTABLE_IMPL_DIR}")

macro(corekit_runtime_bootstrap)

    corekit_bsp_bootstrap()

endmacro()

macro(corekit_runtime_configure)

    corekit_bsp_configure()
    
endmacro()