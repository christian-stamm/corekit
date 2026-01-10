cmake_minimum_required(VERSION 3.19)

include(FetchContent)
# Fetch Fast-CDR library
FetchContent_Declare(
  fastcdr
  GIT_REPOSITORY https://github.com/eProsima/Fast-CDR.git
  GIT_TAG master
)
FetchContent_MakeAvailable(fastcdr)