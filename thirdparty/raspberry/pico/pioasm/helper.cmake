function(pioasm_generate_headers)
    cmake_parse_arguments(
        ARG
        ""
        "LIB_NAME;IMPORT_DIR;EXPORT_DIR"
        "FLAGS"
        ${ARGN}
    )

    file(GLOB_RECURSE programs CONFIGURE_DEPENDS
        "${ARG_IMPORT_DIR}/*.pio"
    )

    set(generated_headers)

    foreach(program IN LISTS programs)
        get_filename_component(filename "${program}" NAME_WE)
        set(generated_include "${ARG_EXPORT_DIR}/${filename}.hpp")

        add_custom_command(
            OUTPUT "${generated_include}"
            COMMAND pioasm ${ARG_FLAGS} "${program}" "${generated_include}"
            DEPENDS "${program}"
            COMMENT "Building ${program}..."
            VERBATIM
        )

        list(APPEND generated_headers "${generated_include}")
    endforeach()

    message(STATUS "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    message(STATUS "Generated headers: ${generated_headers}")

    file(MAKE_DIRECTORY "${ARG_EXPORT_DIR}")

    add_custom_target(
        ${ARG_LIB_NAME}_generate
        DEPENDS ${generated_headers}
    )

    add_library(${ARG_LIB_NAME} INTERFACE)

    target_include_directories(${ARG_LIB_NAME} INTERFACE
        "${ARG_EXPORT_DIR}"
    )

    target_link_libraries(${ARG_LIB_NAME} INTERFACE
        hardware_pio
    )
    
    set(generate_target GENERATE_${ARG_LIB_NAME})
    add_custom_target(${generate_target} ALL
                        DEPENDS ${generated_headers}
                        COMMENT "Generating pioasm target ${ARG_LIB_NAME}")

endfunction()