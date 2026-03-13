cmake_minimum_required(VERSION 3.13)


if(BUILD_COREKIT_PICO)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pioasm)

    pioasm_generate_headers(
        TARGET 
            pixpio
        PROGRAMS
            ${CMAKE_CURRENT_SOURCE_DIR}/pio/pixbus.pio
    )

    # add_dependencies(${PROJECT_NAME}
    #     GENERATE_pixpio
    # )

    # target_link_libraries(${PROJECT_NAME} PRIVATE
    #     corekit
    #     pio
    # )

endif()