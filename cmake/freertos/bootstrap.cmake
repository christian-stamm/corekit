cmake_minimum_required(VERSION 3.25)

if(COREKIT_BASE_OS STREQUAL "freertos")
  
    option(
        COREKIT_BUILD_PLATFORM_FREERTOS
        "Build FreeRTOS Platform"
        ON
    )

    set(FREERTOS_CONFIG_FILE_DIRECTORY ${COREKIT_WORKDIR}/thirdparty/freertos)

    include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)
    include("$ENV{FREERTOS_KERNEL_PATH}/portable/${FREERTOS_PORTABLE_SUBDIR}/FreeRTOS_Kernel_import.cmake")

endif()