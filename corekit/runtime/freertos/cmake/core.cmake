cmake_minimum_required(VERSION 3.25)

macro(corekit_bootstrap)

    corekit_bsp_bootstrap()

    if(NOT EXISTS ${FREERTOS_PORTABLE_IMPL_DIR})
        message(FATAL_ERROR "FREERTOS_PORTABLE_IMPL_DIR does not exist: ${FREERTOS_PORTABLE_IMPL_DIR}. Please ensure that the FreeRTOS portable implementation for the selected Target is available.")
    endif()

    include("${FREERTOS_PORTABLE_IMPL_DIR}/cmake/library.cmake")

endmacro()

macro(corekit_configure)

    corekit_bsp_configure()
    corekit_finalize()
    
endmacro()