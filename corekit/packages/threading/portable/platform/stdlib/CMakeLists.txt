cmake_minimum_required(VERSION 3.25)

corekit_add_module(stoptoken IMPL
    PUBLIC_INCLUDES     "inc"
)

corekit_add_module(executor IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/executor.cpp"
    DEPENDENCIES        core conditionvariable mutex stoptoken 
)