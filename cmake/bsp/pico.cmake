cmake_minimum_required(VERSION 3.25)

###################################
########### BSP Stage 0 ###########
###################################

macro(corekit_bsp_stage_0)

    option(
        COREKIT_BSP_PICO
        "Build PICO Platform"
        ON
    )

    if (NOT EXISTS $ENV{PICO_SDK_PATH})
        message(FATAL_ERROR "No valid PICO_SDK_PATH is set. Scanned path '$ENV{PICO_SDK_PATH}'.")
    endif()

    if(NOT DEFINED PICO_PLATFORM)
        message(FATAL_ERROR "PICO_PLATFORM is not set. Please set PICO_PLATFORM to <rp2040,rp2350-arm-s,risc-v>.")
    endif()

    if(NOT DEFINED PICO_BOARD)
        message(FATAL_ERROR "PICO_BOARD is not set. Please set PICO_BOARD to the desired board. <pico,pico_w,...>.")
    endif()

    set(PICO_CXX_ENABLE_EXCEPTIONS ON)
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
    set(PICO_BOARD_HEADER_DIRS ${COREKIT_WORKDIR}/thirdparty/raspberry/pico/boards)

endmacro()

###################################
########### BSP Stage 1 ###########
###################################

macro(corekit_bsp_stage_1)

    set(PICO_SDK_IMPORT_FILE "${PICO_SDK_PATH}/external/pico_sdk_import.cmake")

    if(NOT EXISTS ${PICO_SDK_IMPORT_FILE})
        message(FATAL_ERROR "PICO_SDK_IMPORT_FILE does not exist: ${PICO_SDK_IMPORT_FILE}. Please ensure that PICO_SDK_PATH is set correctly.")
    endif()

    include(${PICO_SDK_IMPORT_FILE})
    
endmacro()

###################################
########### BSP Stage 2 ###########
###################################

macro(corekit_bsp_stage_2)

    pico_sdk_init()

    if(COREKIT_PAL_FREERTOS)
        if(PICO_PLATFORM STREQUAL "rp2040")
            set(FREERTOS_PORTABLE_IMPL_DIR ${COREKIT_WORKDIR}/thirdparty/freertos/portable/pico/rp2040)
        elseif(PICO_PLATFORM STREQUAL "rp2350-arm-s")
            set(FREERTOS_PORTABLE_IMPL_DIR ${COREKIT_WORKDIR}/thirdparty/freertos/portable/pico/rp2350/arm)
        elseif(PICO_PLATFORM STREQUAL "risc-v")
            set(FREERTOS_PORTABLE_IMPL_DIR ${COREKIT_WORKDIR}/thirdparty/freertos/portable/pico/rp2350/riscv)
        else()
            message(FATAL_ERROR "Unsupported PICO_PLATFORM: ${PICO_PLATFORM}. Supported platforms are: rp2040, rp2350-arm-s, riscv.")
        endif()
         
    endif()
    
endmacro()

###################################
########### BSP Stage 3 ###########
###################################

macro(corekit_bsp_stage_3)

endmacro()
    