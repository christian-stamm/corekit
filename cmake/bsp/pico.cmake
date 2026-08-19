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

    if(COREKIT_PAL_FREERTOS)
        if(
            PICO_BOARD STREQUAL "pico" OR
            PICO_BOARD STREQUAL "pico_w"
        )
            set(PICO_PLATFORM "rp2040")
            set(FREERTOS_PORTABLE_IMPL_DIR ${COREKIT_WORKDIR}/thirdparty/freertos/portable/pico/rp2040)
        elseif(
            PICO_BOARD STREQUAL "bernd" OR
            PICO_BOARD STREQUAL "pico2" OR
            PICO_BOARD STREQUAL "pico2_w"
        )
            set(PICO_PLATFORM "rp2350-arm-s")
            set(FREERTOS_PORTABLE_IMPL_DIR ${COREKIT_WORKDIR}/thirdparty/freertos/portable/pico/rp2350/arm)
        elseif(PICO_BOARD STREQUAL "<unsupported>")
            set(PICO_PLATFORM "rp2350-risc-v")
            set(FREERTOS_PORTABLE_IMPL_DIR ${COREKIT_WORKDIR}/thirdparty/freertos/portable/pico/rp2350/riscv)
        else()
            message(FATAL_ERROR "Unsupported PICO_BOARD: ${PICO_BOARD}. Supported boards are: pico, pico_w, pico2, pico2_w, bernd.")
        endif()
         
    endif()
    
endmacro()

###################################
########### BSP Stage 2 ###########
###################################

macro(corekit_bsp_stage_2)

    pico_sdk_init()
    
endmacro()

###################################
########### BSP Stage 3 ###########
###################################

macro(corekit_bsp_stage_3)

endmacro()
    