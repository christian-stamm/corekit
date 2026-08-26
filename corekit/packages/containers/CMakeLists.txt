cmake_minimum_required(VERSION 3.25)

corekit_add_module(queue API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/queue.cpp"
    DEPENDENCIES        "assert;conditionvariable;mutex"
)

corekit_add_module(stack API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/stack.cpp"
    DEPENDENCIES        "assert;conditionvariable;mutex"
)