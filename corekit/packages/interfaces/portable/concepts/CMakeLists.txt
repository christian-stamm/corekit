cmake_minimum_required(VERSION 3.25)

corekit_add_module(time API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/time.cpp"
)