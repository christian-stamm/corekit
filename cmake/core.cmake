cmake_minimum_required(VERSION 3.25)

message(STATUS "COREKIT_PAL_FILE: ${COREKIT_PAL_FILE}")
message(STATUS "COREKIT_BSP_PATH: ${COREKIT_BSP_PATH}")
message(STATUS "COREKIT_CTEST_ENABLED: ${COREKIT_CTEST_ENABLED}")

if(NOT DEFINED COREKIT_PAL)
    message(FATAL_ERROR "COREKIT_PAL is not set. Choose from: stdlib, freertos, baremetal, zephyr, ...")
endif()

if(EXISTS ${COREKIT_PAL_FILE})
    include(${COREKIT_PAL_FILE})

    if( NOT COMMAND corekit_pal_stage_0 OR 
        NOT COMMAND corekit_pal_stage_1 )
        message(FATAL_ERROR "corekit_pal_stage_<0-3> function are not defined in ${COREKIT_PAL_FILE}. Please ensure that the PAL file defines them.")
    endif()

else()
    message(FATAL_ERROR "Unsupported PAL: ${COREKIT_PAL}. File not found: ${COREKIT_PAL_FILE}")
endif()
    
if(EXISTS ${COREKIT_BSP_FILE})
    include(${COREKIT_BSP_FILE})

    if( NOT COMMAND corekit_bsp_stage_0 OR 
        NOT COMMAND corekit_bsp_stage_1 OR
        NOT COMMAND corekit_bsp_stage_2 OR
        NOT COMMAND corekit_bsp_stage_3 )
        message(FATAL_ERROR "corekit_bsp_stage_<0-3> function are not defined in ${COREKIT_BSP_FILE}. Please ensure that the BSP file defines them.")
    endif()

else()
    message(FATAL_ERROR "Unsupported BSP: ${COREKIT_BSP}. File not found: ${COREKIT_BSP_FILE}")
endif()

if(NOT DEFINED COREKIT_CTEST_ENABLED)
    message(WARNING "COREKIT_CTEST_ENABLED is not set. Disabling tests.")
    option(COREKIT_CTEST_ENABLED "Build Tests" OFF)
endif()

if (COREKIT_CTEST_ENABLED)
    enable_testing()
endif()