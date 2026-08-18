cmake_minimum_required(VERSION 3.25)

if(COREKIT_BUILD_PLATFORM_FREERTOS)

	# FreeRTOS RP2040 port registers its libraries via PICO_SDK_POST_LIST_FILES.
	# Those targets are created only when pico_sdk_init() runs.
	if(PICO_SDK_PATH AND EXISTS "${PICO_SDK_PATH}/pico_sdk_version.cmake" AND
	   (NOT DEFINED PICO_SDK_VERSION_MAJOR OR NOT DEFINED PICO_SDK_VERSION_MINOR OR NOT DEFINED PICO_SDK_VERSION_REVISION))
		include("${PICO_SDK_PATH}/pico_sdk_version.cmake")
	endif()

	pico_sdk_init()

	# Some FreeRTOS + Pico SDK integration paths do not emit this generated
	# header before compile time; provide a deterministic fallback.
	set(_pico_version_h "${CMAKE_BINARY_DIR}/generated/pico_base/pico/version.h")
	if(PICO_SDK_PATH AND EXISTS "${PICO_SDK_PATH}/src/common/pico_base_headers/include/pico/version.h.in" AND NOT EXISTS "${_pico_version_h}")
		file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated/pico_base/pico")
		configure_file(
			"${PICO_SDK_PATH}/src/common/pico_base_headers/include/pico/version.h.in"
			"${_pico_version_h}"
			@ONLY
		)
	endif()

	if(NOT TARGET FreeRTOS-Kernel-Heap4)
		message(FATAL_ERROR "FreeRTOS target 'FreeRTOS-Kernel-Heap4' was not created. Check FREERTOS_KERNEL_PATH, FREERTOS_PORTABLE_SUBDIR, and FreeRTOS kernel import configuration.")
	endif()

endif()