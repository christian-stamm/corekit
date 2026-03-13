function(pioasm_generate_headers)
    # Parse function arguments.
    set(options)
    set(one_value_args
    "TARGET"
    )
    set(multi_value_args
    "PROGRAMS"
    "FLAGS"
    )
    cmake_parse_arguments(
    PARSE_ARGV 0
    PIOASM_GENERATE_HEADERS
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    )

    set(generated_target_dir "${CMAKE_CURRENT_BINARY_DIR}/${PIOASM_GENERATE_HEADERS_TARGET}")
    set(generated_include_dir "${generated_target_dir}")
    set(all_generated_header_files)

    # Create rules to generate the code for each schema.
    foreach(program ${PIOASM_GENERATE_HEADERS_PROGRAMS})
        get_filename_component(filename ${program} NAME_WE)
        set(generated_include "${generated_include_dir}/${filename}.hpp")
        
        add_custom_command(
            OUTPUT ${generated_include}
            COMMAND $<TARGET_FILE:pioasm> ${PIOASM_GENERATE_HEADERS_FLAGS} ${program} ${generated_include}
            DEPENDS pioasm ${program}
            COMMENT "Building ${program}..."
        )
        
        list(APPEND all_generated_header_files ${generated_include})       
    endforeach()

    add_library(${PIOASM_GENERATE_HEADERS_TARGET} INTERFACE)

    target_include_directories(${PIOASM_GENERATE_HEADERS_TARGET} INTERFACE 
        ${generated_target_dir}
    )    
    
    set(generate_target GENERATE_${PIOASM_GENERATE_HEADERS_TARGET})
    add_custom_target(${generate_target} ALL
                        DEPENDS ${all_generated_header_files}
                        COMMENT "Generating pioasm target ${PIOASM_GENERATE_HEADERS_TARGET}")

endfunction()