cmake_minimum_required(VERSION 3.25)

function(corekit_add_module MODULE_NAME)

    cmake_parse_arguments(
        "ARG"                               # prefix
        "API;IMPL"                          # options
        ""                                  # one-value arguments
        "SOURCES;INCLUDES;DEPENDS;TESTS"    # multi-value arguments
        ${ARGN}                             # arguments to parse
    )

    if(NOT MODULE_NAME)
        message(WARNING "Missing required argument: MODULE_NAME")
        return()
    endif()

    if(ARG_API)
        set(lib_flag PUBLIC)

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
        set(lib_flag PRIVATE)

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

    if(NOT lib_flag)
        message(WARNING "Module '${MODULE_NAME}' must specify at least one of API or IMPL")
        return()
    endif()


    get_property(
        existing_src_files
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_SOURCE_FILES"
    )

    get_property(
        existing_test_files
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_TEST_FILES"
    )

    get_property(
        existing_inc_paths
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_${lib_flag}_INC_PATHS"
    )

    get_property(
        existing_dep_libs
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_${lib_flag}_DEP_LIBS"
    )

    set(new_dep_libs ${ARG_DEPENDS})
    make_paths_abs(new_inc_paths "${ARG_INCLUDES}")
    make_paths_abs(new_src_files "${ARG_SOURCES}")
    make_paths_abs(new_test_files "${ARG_TESTS}")

    list(APPEND new_src_files ${existing_src_files})
    list(APPEND new_inc_paths ${existing_inc_paths})
    list(APPEND new_dep_libs ${existing_dep_libs})
    list(APPEND new_test_files ${existing_test_files})

    list(REMOVE_DUPLICATES new_src_files)
    list(REMOVE_DUPLICATES new_inc_paths)
    list(REMOVE_DUPLICATES new_dep_libs)
    list(REMOVE_DUPLICATES new_test_files)

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_SOURCE_FILES"
        "${new_src_files}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_${lib_flag}_INC_PATHS"
        "${new_inc_paths}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_${lib_flag}_DEP_LIBS"
        "${new_dep_libs}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_${MODULE_NAME}_TEST_FILES"
        "${new_test_files}"
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
            pub_inc_paths
            GLOBAL
            PROPERTY "COREKIT_${module}_PUBLIC_INC_PATHS"
        )

        get_property(
            priv_inc_paths
            GLOBAL
            PROPERTY "COREKIT_${module}_PRIVATE_INC_PATHS"
        )

        get_property(
            pub_dep_libs
            GLOBAL
            PROPERTY "COREKIT_${module}_PUBLIC_DEP_LIBS"
        )

        get_property(
            priv_dep_libs
            GLOBAL
            PROPERTY "COREKIT_${module}_PRIVATE_DEP_LIBS"
        )

        get_property(
            test_files
            GLOBAL
            PROPERTY "COREKIT_${module}_TEST_FILES"
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

        unset(matched_api_deps)
        unset(missing_api_deps)
        unset(matched_impl_deps)
        unset(missing_impl_deps)

        match_lib_deps(
            pub_dep_libs
            api_modules
            matched_api_deps
            missing_api_deps
        )

        match_lib_deps(
            priv_dep_libs
            api_modules
            matched_impl_deps
            missing_impl_deps
        )

        message(STATUS "${module} - MATCHED API DEPS: ${matched_api_deps}")
        message(STATUS "${module} - MATCHED IMPL DEPS: ${matched_impl_deps}")

        if(missing_api_deps OR missing_impl_deps)
            message(
                WARNING
                "Module '${module}' requires unavailable dependencies:\n"
                "API: ${missing_api_deps},\n"
                "IMPL: ${missing_impl_deps},\n"
                "Skipping target creation."
            )

            set(can_be_built FALSE)

        endif()

        if(NOT can_be_built)
            continue()
        endif()

        add_library(${target_name} "${target_type}" ${src_files})

        if(target_type STREQUAL "INTERFACE")
            target_include_directories(${target_name} INTERFACE ${pub_inc_paths} ${priv_inc_paths})
            target_link_libraries(${target_name} INTERFACE ${matched_api_deps} ${matched_impl_deps})
        elseif(target_type STREQUAL "STATIC")
            target_include_directories(${target_name} PUBLIC ${pub_inc_paths})
            target_include_directories(${target_name} PUBLIC ${priv_inc_paths})

            target_link_libraries(${target_name} PUBLIC ${matched_api_deps})
            target_link_libraries(${target_name} PRIVATE ${matched_impl_deps})
        endif()

        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${export_dir}"
            LIBRARY_OUTPUT_DIRECTORY "${export_dir}"
            ARCHIVE_OUTPUT_DIRECTORY "${export_dir}"
        )

        add_library("corekit::${module}" ALIAS ${target_name})
        
        
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

    if(remaining_impl_modules)
        message(
            WARNING
            "The following implementation modules were not built because they did not have a corresponding API:\n"
            "${remaining_impl_modules}"
        )

    endif()


endfunction()