cmake_minimum_required(VERSION 3.25)

function(corekit_add_module NAME)
    # ----------------------------------------------------------
    # Pass 1: Create ALL targets
    # ----------------------------------------------------------

    cmake_parse_arguments(
        ARG                             # prefix
        ""                              # options
        ""                              # one-value arguments
        "INCLUDE_PATHS;DEPENDENCIES"    # multi-value arguments
        ${ARGN}                         # arguments to parse
    )

    # Remember module name.
    set_property(
        GLOBAL APPEND
        PROPERTY COREKIT_API_MODULES
        "${NAME}"
    )

    make_paths_abs(includes ${ARG_INCLUDE_PATHS})
    make_paths_abs(tests ${ARG_TEST_FILES})

    set_property(
        GLOBAL APPEND
        PROPERTY "COREKIT_API_${NAME}_INCLUDE_PATHS"
        "${includes}"
    )

    set_property(
        GLOBAL APPEND
        PROPERTY "COREKIT_API_${NAME}_DEPENDENCIES"
        "${ARG_DEPENDENCIES}"
    )

endfunction()

function(corekit_add_impl NAME)
    # ----------------------------------------------------------
    # Pass 1: Create ALL targets
    # ----------------------------------------------------------

    cmake_parse_arguments(
        ARG                                                     # prefix
        ""                                                      # options
        ""                                                      # one-value arguments
        "SOURCE_FILES;INCLUDE_PATHS;DEPENDENCIES;TEST_FILES"    # multi-value arguments
        ${ARGN}                                                 # arguments to parse
    )

    make_paths_abs(sources ${ARG_SOURCE_FILES})
    make_paths_abs(includes ${ARG_INCLUDE_PATHS})
    make_paths_abs(tests ${ARG_TEST_FILES})

    # Remember module name.
    set_property(
        GLOBAL APPEND
        PROPERTY COREKIT_IMPL_MODULES
        "${NAME}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_IMPL_${NAME}_SOURCE_FILES"
        "${sources}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_IMPL_${NAME}_INCLUDE_PATHS"
        "${includes}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_IMPL_${NAME}_DEPENDENCIES"
        "${ARG_DEPENDENCIES}"
    )

    set_property(
        GLOBAL
        PROPERTY "COREKIT_IMPL_${NAME}_TEST_FILES"
        "${tests}"
    )

endfunction()


function(corekit_finalize)
    # ----------------------------------------------------------
    # Pass 2: Configure targets
    #
    # At this point ALL corekit::* targets exist, so dep
    # ordering is irrelevant.
    # ----------------------------------------------------------

    get_property(
        api_modules
        GLOBAL
        PROPERTY COREKIT_API_MODULES
    )

    get_property(
        impl_modules
        GLOBAL
        PROPERTY COREKIT_IMPL_MODULES
    )

    foreach(module IN LISTS api_modules)

        get_property(
            api_deps
            GLOBAL
            PROPERTY "COREKIT_API_${NAME}_DEPENDENCIES"
        )

        get_property(
            api_incs
            GLOBAL
            PROPERTY "COREKIT_API_${NAME}_INCLUDE_PATHS"
        )

        set(missing_deps "")
        set(corekit_libs "")

        foreach(dep IN LISTS api_deps)

            if(dep IN_LIST api_modules)
                list(APPEND ${corekit_libs} "corekit::${dep}")
            else()
                list(APPEND ${missing_deps} "corekit::${dep}")
            endif()

        endforeach()

        if(missing_deps)
            list(JOIN missing_deps "\n - " missing_deps_pretty)

            message(
                WARNING
                "corekit module '${module}' is not available due to missing dependencies:\n"
                " - ${missing_deps_pretty}"
            )

            continue()

        endif()

        add_library("corekit-${module}" INTERFACE)
        target_include_directories("corekit-${module}" INTERFACE ${api_incs})
        target_link_libraries("corekit-${module}" INTERFACE ${corekit_libs})
        add_library("corekit::${module}" ALIAS "corekit-${module}")

    endforeach()

    foreach(module IN LISTS impl_modules)

        if(NOT TARGET "corekit::${module}")
            message(
                WARNING
                "No API available for corekit module '${module}'. Skipping target creation."
            )

            continue()
        endif()

        get_property(
            sources
            GLOBAL
            PROPERTY "COREKIT_${module}_SOURCES"
        )

        get_property(
            includes
            GLOBAL
            PROPERTY "COREKIT_${module}_INCLUDES"
        )

        get_property(
            depends
            GLOBAL
            PROPERTY "COREKIT_${module}_DEPENDS"
        )

        get_property(
            tests
            GLOBAL
            PROPERTY "COREKIT_${module}_TESTS"
        )

        
        set(missing_deps "")
        set(corekit_libs "")

        foreach(dep IN LISTS depends)
            if(NOT "${dep}" MATCHES "^corekit::")
                
            endif()

            if(NOT dep IN_LIST modules)
                list(APPEND missing_deps "${dep}")
            endif()
        endforeach()

        if(missing_deps)
            list(JOIN missing_deps "\n - " missing_deps_pretty)

            message(
                WARNING
                "corekit module '${module}' is not available due to missing dependencies:\n"
                " - ${missing_deps_pretty}"
            )

            continue()

        endif()

        set(lib_name "corekit-${module}-impl")
        set(lib_type INTERFACE)
        set(lib_link INTERFACE)

        if(sources)
            set(lib_type STATIC)
            set(lib_link PUBLIC)
        endif()


        add_library(${lib_name} ${lib_type} ${sources})

        target_include_directories(${lib_name} ${lib_link} ${includes})

        foreach(dep IN LISTS depends)

            if(dep IN_LIST modules)
                set(dep "corekit::${dep}")
            endif()

            target_link_libraries(${lib_name} ${lib_link} "${dep}")

        endforeach()


        add_library("corekit::${module}::impl" ALIAS "${lib_name}")

        string(TOUPPER "${module}" module_upper)
        add_compile_definitions("COREKIT_HAS_${module_upper}=1")

        find_package(GTest QUIET)

        if(EXISTS "${tests}" AND GTest_FOUND)

            add_executable("corekit-${module}-test" ${tests})
            target_link_libraries("corekit-${module}-test" PUBLIC 
                "corekit::${module}"
                GTest::gtest 
                GTest::gtest_main
            )

        endif()

    endforeach()
endfunction()