cmake_minimum_required(VERSION 3.26)
project(spike C)

set(CMAKE_C_STANDARD 23)

include(FetchContent)

FetchContent_Declare(
    v1math
    GIT_REPOSITORY https://github.com/kraglik/v1math.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(v1math)
if(NOT v1math_POPULATED)
    FetchContent_Populate(v1math)
    add_subdirectory(${v1math_SOURCE_DIR} ${v1math_BINARY_DIR})
endif()

find_package(SDL2 REQUIRED)

include_directories(
    include
    ${SDL2_INCLUDE_DIRS}
    ${v1math_INCLUDE_DIRS}
)

add_executable(spike src/main.c)
add_subdirectory(src)

target_link_libraries(spike
    PUBLIC
        ${SDL2_LIBRARIES} v1math)
target_include_directories(spike
    PUBLIC
        ${v1math_SOURCE_DIR}/include)
