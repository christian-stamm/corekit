cmake_minimum_required(VERSION 3.25)

function(corekit_add_module MODULE_NAME)

    cmake_parse_arguments(
        ARG                                 # prefix
        "API;IMPL"                          # options
        ""                                  # one-value arguments
        "SOURCES;INCLUDES;DEPENDS;TESTS"    # multi-value arguments
        ${ARGN}                             # arguments to parse
    )

    if(NOT MODULE_NAME)
        message(WARNING "Missing required argument: MODULE_NAME")
        return()
    endif()

    if(NOT ARG_API AND NOT ARG_IMPL)
        message(WARNING "Module '${MODULE_NAME}' must specify at least one of API or IMPL")
        return()
    endif()

    get_property(
        modules
        GLOBAL
        PROPERTY COREKIT_MODULE_NAMES
    )

    if(NOT "${MODULE_NAME}" IN_LIST modules)
        set_property(
            GLOBAL APPEND
            PROPERTY COREKIT_MODULE_NAMES
            "${MODULE_NAME}"
        )
    endif()

    get_property(
        existing_source_files
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_SOURCE_FILES"
    )

    get_property(
        existing_include_paths
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_INCLUDE_PATHS"
    )

    get_property(
        existing_dependencies
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_DEPENDENCIES"
    )

    get_property(
        existing_test_files
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_TEST_FILES"
    )
    
    set(new_dependencies ${ARG_DEPENDS})
    make_paths_abs(new_source_files "${ARG_SOURCES}")
    make_paths_abs(new_include_paths "${ARG_INCLUDES}")
    make_paths_abs(new_test_files "${ARG_TESTS}")

    list(APPEND new_source_files ${existing_source_files})
    list(APPEND new_include_paths ${existing_include_paths})
    list(APPEND new_dependencies ${existing_dependencies})
    list(APPEND new_test_files ${existing_test_files})

    list(REMOVE_DUPLICATES new_source_files)
    list(REMOVE_DUPLICATES new_include_paths)
    list(REMOVE_DUPLICATES new_dependencies)
    list(REMOVE_DUPLICATES new_test_files)

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_SOURCE_FILES"
        "${new_source_files}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_INCLUDE_PATHS"
        "${new_include_paths}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_DEPENDENCIES"
        "${new_dependencies}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_TEST_FILES"
        "${new_test_files}"
    )

    if(ARG_API)
        set_property(
            GLOBAL
            PROPERTY "COREKIT_${MODULE_NAME}_HAS_API"
            TRUE
        )
    endif()

    if(ARG_IMPL)
        set_property(
            GLOBAL
            PROPERTY "COREKIT_${MODULE_NAME}_HAS_IMPL"
            TRUE
        )

        set_property(
            GLOBAL
            PROPERTY "COREKIT_${MODULE_NAME}_BINARY_DIR"
            "${CMAKE_CURRENT_BINARY_DIR}"
        )
    endif()

endfunction()

function(corekit_build_modules)
    find_package(GTest QUIET)

    get_property(
        modules
        GLOBAL
        PROPERTY COREKIT_MODULE_NAMES
    )

    list(REMOVE_DUPLICATES modules)

    # ----------------------------------------------------------
    # Pass 1: Create all targets
    # ----------------------------------------------------------

    foreach(module IN LISTS modules)
    
        get_property(
            src_files
            GLOBAL
            PROPERTY "COREKIT_${module}_SOURCE_FILES"
        )

        get_property(
            inc_paths
            GLOBAL
            PROPERTY "COREKIT_${module}_INCLUDE_PATHS"
        )

        get_property(
            lib_deps
            GLOBAL
            PROPERTY "COREKIT_${module}_DEPENDENCIES"
        )

        get_property(
            test_files
            GLOBAL
            PROPERTY "COREKIT_${module}_TEST_FILES"
        )

        get_property(
            has_api
            GLOBAL
            PROPERTY "COREKIT_${module}_HAS_API"
        )

        get_property(
            has_impl
            GLOBAL
            PROPERTY "COREKIT_${module}_HAS_IMPL"
        )

        set(is_buildable TRUE)

        if(NOT has_api OR NOT has_impl)
            message(
                WARNING
                "Module '${module}' must specify both API and IMPL to be built. Skipping target creation."
            )

            set(is_buildable FALSE)

        endif()

        set(target_name "corekit-${module}")

        if(src_files)
            set(target_type STATIC)
            set(target_link PUBLIC)
        else()
            set(target_type INTERFACE)
            set(target_link INTERFACE)
        endif()

        set(missing_deps "")
        set(matched_deps "")

        foreach(dep IN LISTS lib_deps)
            if(dep IN_LIST modules)
                # Internal module name, such as atomic, mutex, semaphore, etc.
                list(APPEND matched_deps "corekit::${dep}")
            elseif(TARGET "${dep}")
                # External library name, such as pthread, fmt::fmt,
                # Threads::Threads, etc.
                list(APPEND matched_deps "${dep}")
            else()
                # Dependency is missing, so we will skip this module.
                list(APPEND missing_deps "${dep}")
            endif()
        endforeach()

        if(missing_deps)
            message(
                WARNING
                "Module '${module}' requires unavailable dependencies: "
                "${missing_deps}"
                "Skipping target creation."
            )

            set(is_buildable FALSE)

        endif()

        if(NOT is_buildable)
            continue()
        endif()

        add_library(${target_name} "${target_type}" ${src_files})
        target_include_directories(${target_name} ${target_link} ${inc_paths})
        target_link_libraries(${target_name} ${target_link} ${matched_deps})
        add_library("corekit::${module}" ALIAS ${target_name})

        get_property(
            module_binary_dir
            GLOBAL
            PROPERTY "COREKIT_${module}_BINARY_DIR"
        )

        set(export_dir "${module_binary_dir}/build/${module}")

        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${export_dir}"
            LIBRARY_OUTPUT_DIRECTORY "${export_dir}"
            ARCHIVE_OUTPUT_DIRECTORY "${export_dir}"
        )

        if(GTest_FOUND AND test_files)

            include(GoogleTest)

            add_executable("${target_name}-test" ${test_files})

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