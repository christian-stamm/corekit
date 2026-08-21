cmake_minimum_required(VERSION 3.25)

corekit_register_impl(atomic ${CMAKE_CURRENT_SOURCE_DIR} "")
corekit_register_impl(mutex ${CMAKE_CURRENT_SOURCE_DIR} "")
corekit_register_impl(semaphore ${CMAKE_CURRENT_SOURCE_DIR} "")
corekit_register_impl(stoptoken ${CMAKE_CURRENT_SOURCE_DIR} "")
corekit_register_impl(thread ${CMAKE_CURRENT_SOURCE_DIR} "")
