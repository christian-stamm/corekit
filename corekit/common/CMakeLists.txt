cmake_minimum_required(VERSION 3.25)

corekit_register_api(atomic ${CMAKE_CURRENT_SOURCE_DIR})
corekit_register_api(mutex ${CMAKE_CURRENT_SOURCE_DIR})

if (TARGET corekit::atomic AND TARGET corekit::mutex)
    corekit_register_impl(task ${CMAKE_CURRENT_SOURCE_DIR} "corekit::atomic;corekit::mutex")
    corekit_register_api(task ${CMAKE_CURRENT_SOURCE_DIR})
endif()
