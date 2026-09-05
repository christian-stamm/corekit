cmake_minimum_required(VERSION 3.25)

corekit_add_module(queue API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/queue.cpp"
)

corekit_add_module(stack API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/stack.cpp"
)