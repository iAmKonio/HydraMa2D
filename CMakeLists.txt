cmake_minimum_required(VERSION 3.16)
project(HydraMa2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SOURCES src/*.cpp)

add_executable(HydraMa2D ${SOURCES})
target_compile_options(HydraMa2D PRIVATE -O3 -Wall -Wextra -pedantic)
