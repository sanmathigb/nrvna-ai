cmake_minimum_required(VERSION 3.16)
project(nrvna_ai LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Add llama.cpp submodule
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp)

# Core library sources
set(NRVNA_SOURCES
    src/logger.cpp
    src/work.cpp
    src/scanner.cpp
    src/pool.cpp
    src/processor.cpp
    src/server.cpp
    src/flow.cpp
    src/runner.cpp
)

# Core library
add_library(nrvna_core STATIC ${NRVNA_SOURCES})
target_include_directories(nrvna_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
)
target_link_libraries(nrvna_core PUBLIC 
    llama 
    Threads::Threads
)
target_compile_options(nrvna_core PRIVATE -Wall -Wextra -Wpedantic)

# CLI executables
add_executable(nrvnad cli/nrvnad.cpp)
target_link_libraries(nrvnad nrvna_core)

add_executable(wrk cli/wrk.cpp)
target_link_libraries(wrk nrvna_core)

add_executable(flw cli/flw.cpp)
target_link_libraries(flw nrvna_core)

# Agent - autonomous ReAct loop
add_executable(agent cli/agent-main.cpp)
target_link_libraries(agent nrvna_core)

# Agent with tools - bash execution, web, search
add_executable(agent-tools cli/agent-tools.cpp)
target_link_libraries(agent-tools nrvna_core)
