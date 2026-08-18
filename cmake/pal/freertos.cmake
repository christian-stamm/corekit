cmake_minimum_required(VERSION 3.25)

###################################
########### PAL Stage 0 ###########
###################################

macro(corekit_pal_stage_0)

    option(
        COREKIT_PAL_FREERTOS
        "Build Corekit using FreeRTOS backend"
        ON
    )

    set(FREERTOS_KERNEL_PATH $ENV{FREERTOS_KERNEL_PATH})
    set(FREERTOS_CONFIG_FILE_DIRECTORY ${COREKIT_WORKDIR}/thirdparty/freertos/common)

endmacro()

###################################
########### PAL Stage 1 ###########
###################################

macro(corekit_pal_stage_1)

    if(NOT DEFINED FREERTOS_PORTABLE_IMPL_DIR)
		message(FATAL_ERROR "FREERTOS_PORTABLE_IMPL_DIR is not set. BSP needs to set FREERTOS_PORTABLE_IMPL_DIR.")
	endif()

	set(FREERTOS_PORTABLE_CMAKE_FILE "${FREERTOS_PORTABLE_IMPL_DIR}/cmake/library.cmake")

	if(EXISTS ${FREERTOS_PORTABLE_CMAKE_FILE})
		include(${FREERTOS_PORTABLE_CMAKE_FILE})
	else()
		message(FATAL_ERROR "Incomplete FreeRTOS portable implementation: ${FREERTOS_PORTABLE_IMPL_DIR} does not contain a './cmake/library.cmake' file. 
		Please ensure that the FreeRTOS portable can be configured based on ${FREERTOS_PORTABLE_CMAKE_FILE}.")
	endif()

endmacro()


