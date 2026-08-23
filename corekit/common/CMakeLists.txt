cmake_minimum_required(VERSION 3.25)

corekit_add_module(task SOURCES src/task.cpp DEPENDS atomic stoptoken)