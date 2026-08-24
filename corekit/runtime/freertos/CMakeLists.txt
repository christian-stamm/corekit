cmake_minimum_required(VERSION 3.25)
set(FREERTOS_KERNEL_PATH "$ENV{FREERTOS_KERNEL_PATH}" CACHE INTERNAL "")
set(COREKIT_FREERTOS_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "")
set(FREERTOS_MODULES_IMPL_DIR "${COREKIT_FREERTOS_DIR}/modules" CACHE INTERNAL "")
set(FREERTOS_PORTABLE_IMPL_DIR "${COREKIT_FREERTOS_DIR}/portable/${COREKIT_TARGET}" CACHE INTERNAL "")

macro(corekit_runtime_bootstrap)

    set(FREERTOS_PORTABLE_IMPL_DIR "${COREKIT_FREERTOS_DIR}/portable/${COREKIT_TARGET}")
    add_subdirectory("${FREERTOS_PORTABLE_IMPL_DIR}")
    
    if(COMMAND corekit_bsp_bootstrap)
        corekit_bsp_bootstrap()
    endif()

endmacro()

macro(corekit_runtime_configure)

    if(COMMAND corekit_bsp_configure)
        corekit_bsp_configure()
    endif()

    set(FREERTOS_MODULES_IMPL_DIR "${COREKIT_FREERTOS_DIR}/modules")
    add_subdirectory("${FREERTOS_MODULES_IMPL_DIR}")

endmacro()