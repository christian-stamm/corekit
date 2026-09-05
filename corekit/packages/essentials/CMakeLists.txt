cmake_minimum_required(VERSION 3.25)

corekit_add_module(core API IMPL
    PUBLIC_INCLUDES     "inc"
)

corekit_add_module(check API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/check.cpp"
    DEPENDENCIES        "error;logger;mutex"
)

