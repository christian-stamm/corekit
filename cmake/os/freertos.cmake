cmake_minimum_required(VERSION 3.25)


function(corekit_bootstrap bootstrap_file)

    option(
        COREKIT_BUILD_FREERTOS
        "Build FreeRTOS Platform"
        ON
    )

    set(FREERTOS_KERNEL_PATH $ENV{FREERTOS_KERNEL_PATH})
    set(FREERTOS_CONFIG_FILE_DIRECTORY ${COREKIT_WORKDIR}/thirdparty/freertos/common)

    include(${bootstrap_file})

endfunction()


function(corekit_configure configure_file)

    include(${configure_file})

    if(NOT DEFINED FREERTOS_PORTABLE_IMPL_DIR)
		message(FATAL_ERROR "FREERTOS_PORTABLE_IMPL_DIR is not set. Please set it to the path of the FreeRTOS portable implementation directory.")
	endif()

	set(FREERTOS_PORTABLE_CMAKE_FILE "${FREERTOS_PORTABLE_IMPL_DIR}/cmake/library.cmake")

	if(EXISTS ${FREERTOS_PORTABLE_CMAKE_FILE})
		include(${FREERTOS_PORTABLE_CMAKE_FILE})
	else()
		message(FATAL_ERROR "Incomplete FreeRTOS portable implementation: ${FREERTOS_PORTABLE_IMPL_DIR} does not contain a cmake/library.cmake file. 
		Please ensure that the FreeRTOS portable can be configured based on ${FREERTOS_PORTABLE_CMAKE_FILE}.")
	endif()

endfunction()


