cmake_minimum_required(VERSION 3.25)

corekit_add_module(time API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/time.cpp"
)

corekit_add_module(uartdevice API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/uartdevice.cpp"
    DEPENDENCIES        "serialdevice"
)

corekit_add_module(spidevice API
    PUBLIC_INCLUDES     "inc"
    SOURCE_FILES        "src/spidevice.cpp"
    DEPENDENCIES        "serialdevice"
)

# corekit_add_module(i2cdevice API
#     PUBLIC_INCLUDES     "inc"
#     SOURCE_FILES        "src/i2cdevice.cpp"
#     DEPENDENCIES        "serialdevice"
# )

# corekit_add_module(piodevice API
#     PUBLIC_INCLUDES     "inc"
#     SOURCE_FILES        "src/piodevice.cpp"
#     DEPENDENCIES        "serialdevice"
# )

# corekit_add_module(dmacontroller API
#     PUBLIC_INCLUDES     "inc"
#     SOURCE_FILES        "src/dmacontroller.cpp"
#     DEPENDENCIES        "serialdevice"
# )