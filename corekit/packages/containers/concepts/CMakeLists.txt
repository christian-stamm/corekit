cmake_minimum_required(VERSION 3.25)

corekit_add_module(queue API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/queue.cpp"
    DEPENDENCIES        "result"
)

corekit_add_module(stack API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/stack.cpp"
    DEPENDENCIES        "result"
)

corekit_add_module(ringbuffer API IMPL
    PUBLIC_INCLUDES     "inc"
    DEPENDENCIES        "mutex;result"
)

corekit_add_module(doublebuffer API IMPL
    PUBLIC_INCLUDES     "inc"
    DEPENDENCIES        "mutex;result"
)