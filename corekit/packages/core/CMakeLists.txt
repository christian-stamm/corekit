cmake_minimum_required(VERSION 3.25)

corekit_add_module(core API IMPL
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/assert.cpp"
)