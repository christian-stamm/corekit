cmake_minimum_required(VERSION 3.19)

include(FetchContent)
# Fetch thread-pool library
FetchContent_Declare(
  threadpool
  GIT_REPOSITORY https://github.com/DeveloperPaul123/thread-pool.git
  GIT_TAG main
)
FetchContent_MakeAvailable(threadpool)