cmake_minimum_required(VERSION 3.25)

corekit_add_module(core API IMPL
    INCLUDES   "inc"
    SOURCES    "src/assert.cpp"
)