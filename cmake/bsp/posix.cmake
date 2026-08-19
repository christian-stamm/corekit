cmake_minimum_required(VERSION 3.25)

###################################
########### BSP Stage 0 ###########
###################################

macro(corekit_bsp_stage_0)

endmacro()

###################################
########### BSP Stage 1 ###########
###################################

macro(corekit_bsp_stage_1)

endmacro()

###################################
########### BSP Stage 2 ###########
###################################

macro(corekit_bsp_stage_2)
    
    if(COREKIT_PAL_FREERTOS)

        set(FREERTOS_PORTABLE_IMPL_DIR "${COREKIT_WORKDIR}/thirdparty/freertos/portable/posix")
       
    endif()

endmacro()

###################################
########### BSP Stage 3 ###########
###################################

macro(corekit_bsp_stage_3)
    
endmacro()
    