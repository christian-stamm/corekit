cmake_minimum_required(VERSION 3.25)

corekit_add_module(heapbuffer
    INCLUDES     "inc"
    DEPENDENCIES "check;math;memory"
)

corekit_add_module(ringbuffer
    INCLUDES     "inc"
    DEPENDENCIES "check;math;mutex"
)

corekit_add_module(doublebuffer
    INCLUDES     "inc"
    DEPENDENCIES "check;heapbuffer"
)