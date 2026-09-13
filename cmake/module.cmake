cmake_minimum_required(VERSION 3.25)

function(corekit_add_module MODULE_NAME)

    cmake_parse_arguments(
        "ARG"                                                   # prefix
        ""                                                      # options
        ""                                                      # one-value arguments
        "INCLUDES;SOURCES;DEPENDENCIES;TESTS"                   # multi-value arguments
        ${ARGN}                                                 # arguments to parse
    )

    if(NOT MODULE_NAME)
        message(WARNING "Missing required argument: MODULE_NAME")
        return()
    endif()

    get_property(
        modules
        GLOBAL
        PROPERTY COREKIT_MODULES
    )

    if(MODULE_NAME IN_LIST modules)
        message(WARNING "Module '${MODULE_NAME}' has already been registered.")
        return()
    endif()

    set_property(
        GLOBAL APPEND
        PROPERTY COREKIT_MODULES
        ${MODULE_NAME}
    )
        
    make_paths_abs(includes "${ARG_INCLUDES}")
    make_paths_abs(sources "${ARG_SOURCES}")
    make_paths_abs(tests "${ARG_TESTS}")
    set(deps ${ARG_DEPENDENCIES})

    list(REMOVE_DUPLICATES includes)
    list(REMOVE_DUPLICATES sources)
    list(REMOVE_DUPLICATES tests)
    list(REMOVE_DUPLICATES deps)

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_INCLUDES"
        "${includes}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_SOURCES"
        "${sources}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_TESTS"
        "${tests}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_DEPS"
        "${deps}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_BIN_DIR"
        "${CMAKE_CURRENT_BINARY_DIR}"
    )

endfunction()

function(corekit_build_modules)
    find_package(GTest QUIET)

    get_property(
        modules
        GLOBAL
        PROPERTY COREKIT_MODULES
    )

    list(REMOVE_DUPLICATES modules)

    # ----------------------------------------------------------
    # Pass 1: Create all targets
    # ----------------------------------------------------------

    foreach(module IN LISTS modules)

        get_property(
            includes
            GLOBAL
            PROPERTY "COREKIT_${module}_INCLUDES"
        )
    
        get_property(
            sources
            GLOBAL
            PROPERTY "COREKIT_${module}_SOURCES"
        )

        get_property(
            tests
            GLOBAL
            PROPERTY "COREKIT_${module}_TESTS"
        )

        get_property(
            deps
            GLOBAL
            PROPERTY "COREKIT_${module}_DEPS"
        )

        get_property(
            bin_dir
            GLOBAL
            PROPERTY "COREKIT_${module}_BIN_DIR"
        )

        set(can_be_built TRUE)
        set(target_name "corekit-${module}")
        set(export_dir "${bin_dir}/build/${module}")

        if(sources)
            set(target_type STATIC)
        else()
            set(target_type INTERFACE)
        endif()

        unset(matched_deps)
        unset(missing_deps)

        resolve_dependencies(
            deps
            modules
            matched_deps
            missing_deps
        )

        if(missing_deps)
            message(
                WARNING
                "Module '${module}' requires unavailable dependencies:\n"
                "${missing_deps},\n"
                "Skipping target creation."
            )

            set(can_be_built FALSE)

        endif()

        if(NOT can_be_built)
            continue()
        endif()

        add_library(${target_name} ${target_type} ${sources})

        if(target_type STREQUAL "INTERFACE")
            target_include_directories(${target_name} INTERFACE ${includes})
            target_link_libraries(${target_name} INTERFACE ${matched_deps})
        elseif(target_type STREQUAL "STATIC")
            target_include_directories(${target_name} PUBLIC ${includes})
            target_link_libraries(${target_name} PUBLIC ${matched_deps})
        endif()

        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${export_dir}"
            LIBRARY_OUTPUT_DIRECTORY "${export_dir}"
            ARCHIVE_OUTPUT_DIRECTORY "${export_dir}"
        )

        add_library("corekit::${module}" ALIAS ${target_name})
        
        
        if(GTest_FOUND AND tests)

            include(GoogleTest)

            add_executable("${target_name}-test" ${tests})

            set_target_properties("${target_name}-test" PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${export_dir}"
                LIBRARY_OUTPUT_DIRECTORY "${export_dir}"
                ARCHIVE_OUTPUT_DIRECTORY "${export_dir}"
            )

            target_link_libraries("${target_name}-test" PRIVATE 
                ${target_name}
                GTest::gtest 
                GTest::gtest_main
            )

            gtest_add_tests(TARGET "${target_name}-test")

        endif()

    endforeach()

endfunction()