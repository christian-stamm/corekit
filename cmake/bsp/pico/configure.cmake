cmake_minimum_required(VERSION 3.25)

if(COREKIT_BUILD_BSP_PICO)

    pico_sdk_init()

    if(COREKIT_BUILD_FREERTOS)
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

endif()