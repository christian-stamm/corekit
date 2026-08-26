cmake_minimum_required(VERSION 3.25)

function(corekit_add_module MODULE_NAME)

    cmake_parse_arguments(
        "ARG"                                                                       # prefix
        "API;IMPL"                                                                  # options
        ""                                                                          # one-value arguments
        "SOURCE_FILES;PUBLIC_INCLUDES;PRIVATE_INCLUDES;DEPENDENCIES;GTEST_FILES"    # multi-value arguments
        ${ARGN}                                                                     # arguments to parse
    )

    if(NOT MODULE_NAME)
        message(WARNING "Missing required argument: MODULE_NAME")
        return()
    endif()

    if(NOT ARG_API AND NOT ARG_IMPL)
        message(WARNING "Module '${MODULE_NAME}' must specify at least one of API or IMPL")
        return()
    endif()

    if(ARG_API)
        get_property(
            api_modules
            GLOBAL
            PROPERTY COREKIT_MODULE_APIS
        )

        if(MODULE_NAME IN_LIST api_modules)
            message(WARNING "Module '${MODULE_NAME}' has already been registered as an API module")
            return()
        endif()

        set_property(
            GLOBAL APPEND
            PROPERTY "COREKIT_MODULE_APIS"
            ${MODULE_NAME}
        )
        
    endif()

    if(ARG_IMPL)

        get_property(
            impl_modules
            GLOBAL
            PROPERTY COREKIT_MODULE_IMPLS
        )

        if(MODULE_NAME IN_LIST impl_modules)
            message(WARNING "Module '${MODULE_NAME}' has already been registered as an implementation module")
            return()
        endif()

        set_property(
            GLOBAL APPEND
            PROPERTY "COREKIT_MODULE_IMPLS"
            ${MODULE_NAME}
        )
        
        set_property(
            GLOBAL
            PROPERTY "COREKIT_${MODULE_NAME}_BINARY_DIR"
            "${CMAKE_CURRENT_BINARY_DIR}"
        )
        
    endif()

    get_property(
        existing_source_files
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_SOURCE_FILES"
    )

    get_property(
        existing_gtest_files
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_GTEST_FILES"
    )

    get_property(
        existing_public_includes
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_PUBLIC_INCLUDES"
    )

    get_property(
        existing_private_includes
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_PRIVATE_INCLUDES"
    )

    get_property(
        existing_dependencies
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_DEPENDENCIES"
    )

    make_paths_abs(new_public_includes "${ARG_PUBLIC_INCLUDES}")
    make_paths_abs(new_private_includes "${ARG_PRIVATE_INCLUDES}")
    make_paths_abs(new_source_files "${ARG_SOURCE_FILES}")
    make_paths_abs(new_gtest_files "${ARG_GTEST_FILES}")
    set(new_dependencies ${ARG_DEPENDENCIES})

    list(APPEND new_source_files ${existing_source_files})
    list(APPEND new_public_includes ${existing_public_includes})
    list(APPEND new_private_includes ${existing_private_includes})
    list(APPEND new_dependencies ${existing_dependencies})
    list(APPEND new_gtest_files ${existing_gtest_files})

    list(REMOVE_DUPLICATES new_source_files)
    list(REMOVE_DUPLICATES new_public_includes)
    list(REMOVE_DUPLICATES new_private_includes)
    list(REMOVE_DUPLICATES new_dependencies)
    list(REMOVE_DUPLICATES new_gtest_files)

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_SOURCE_FILES"
        "${new_source_files}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_PUBLIC_INCLUDES"
        "${new_public_includes}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_PRIVATE_INCLUDES"
        "${new_private_includes}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_DEPENDENCIES"
        "${new_dependencies}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_GTEST_FILES"
        "${new_gtest_files}"
    )

endfunction()

function(corekit_build_modules)
    find_package(GTest QUIET)

    get_property(
        api_modules
        GLOBAL
        PROPERTY COREKIT_MODULE_APIS
    )

    get_property(
        impl_modules
        GLOBAL
        PROPERTY COREKIT_MODULE_IMPLS
    )

    list(REMOVE_DUPLICATES api_modules)
    list(REMOVE_DUPLICATES impl_modules)

    set(remaining_impl_modules ${impl_modules})

    # ----------------------------------------------------------
    # Pass 1: Create all targets
    # ----------------------------------------------------------

    foreach(module IN LISTS api_modules)
    
        get_property(
            src_files
            GLOBAL
            PROPERTY "COREKIT_${module}_SOURCE_FILES"
        )

        get_property(
            public_includes
            GLOBAL
            PROPERTY "COREKIT_${module}_PUBLIC_INCLUDES"
        )

        get_property(
            private_includes
            GLOBAL
            PROPERTY "COREKIT_${module}_PRIVATE_INCLUDES"
        )

        get_property(
            dependencies
            GLOBAL
            PROPERTY "COREKIT_${module}_DEPENDENCIES"
        )

        get_property(
            gtest_files
            GLOBAL
            PROPERTY "COREKIT_${module}_GTEST_FILES"
        )

        get_property(
            bin_dir
            GLOBAL
            PROPERTY "COREKIT_${module}_BINARY_DIR"
        )


        set(can_be_built TRUE)
        set(target_name "corekit-${module}")
        set(export_dir "${bin_dir}/build/${module}")

        if(src_files)
            set(target_type STATIC)
        else()
            set(target_type INTERFACE)
        endif()

        if(NOT module IN_LIST impl_modules)

            message(
                WARNING
                "Module '${module}' must specify both API and IMPL to be built. Skipping target creation."
            )

            set(can_be_built FALSE)
        else()
            list(REMOVE_ITEM remaining_impl_modules ${module})
        endif()

        unset(matched_deps)
        unset(missing_deps)

        resolve_dependencies(
            dependencies
            api_modules
            matched_deps
            missing_deps
        )

        message(STATUS "${module} - MATCHED DEPS: ${matched_deps}")
        message(STATUS "${module} - MISSING DEPS: ${missing_deps}")

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

        add_library(${target_name} "${target_type}" ${src_files})

        if(target_type STREQUAL "INTERFACE")
            target_include_directories(${target_name} INTERFACE ${public_includes} ${private_includes})
            target_link_libraries(${target_name} INTERFACE ${matched_deps})
        elseif(target_type STREQUAL "STATIC")
            target_include_directories(${target_name} PUBLIC ${public_includes})
            target_include_directories(${target_name} PRIVATE ${private_includes})
            target_link_libraries(${target_name} PRIVATE ${matched_deps})
        endif()

        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${export_dir}"
            LIBRARY_OUTPUT_DIRECTORY "${export_dir}"
            ARCHIVE_OUTPUT_DIRECTORY "${export_dir}"
        )

        add_library("corekit::${module}" ALIAS ${target_name})
        
        
        if(GTest_FOUND AND gtest_files)

            include(GoogleTest)

            add_executable("${target_name}-test" ${gtest_files})

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

    if(remaining_impl_modules)
        message(
            WARNING
            "The following implementation modules were not built because they did not have a corresponding API:\n"
            "${remaining_impl_modules}"
        )

    endif()


endfunction()