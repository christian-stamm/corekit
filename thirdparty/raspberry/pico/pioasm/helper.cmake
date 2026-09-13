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

        set(generated_include
            "${ARG_EXPORT_DIR}/pioasm/${filename}.h"
        )

        add_custom_command(
            OUTPUT "${generated_include}"
            COMMAND pioasm
                    ${ARG_FLAGS}
                    "${program}"
                    "${generated_include}"
            DEPENDS "${program}"
            COMMENT "Generating ${generated_include}"
            VERBATIM
        )

        list(APPEND generated_headers "${generated_include}")
    endforeach()

    add_custom_target(
        ${ARG_LIB_NAME}_generate
        DEPENDS ${generated_headers}
    )

    add_library(${ARG_LIB_NAME} INTERFACE)

    add_dependencies(
        ${ARG_LIB_NAME}
        ${ARG_LIB_NAME}_generate
    )

    target_include_directories(
        ${ARG_LIB_NAME}
        INTERFACE
        "${ARG_EXPORT_DIR}"
    )

    target_link_libraries(
        ${ARG_LIB_NAME}
        INTERFACE
        hardware_pio
    )
endfunction()