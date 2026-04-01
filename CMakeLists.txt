cmake_minimum_required(VERSION 3.16)
project(nrvna_ai VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Add llama.cpp submodule
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_NUMBER 0 CACHE STRING "" FORCE)
set(LLAMA_INSTALL_VERSION "0.0.0")
add_subdirectory(third_party/llama.cpp)

# Add mtmd library for multimodal support
add_subdirectory(third_party/llama.cpp/tools/mtmd)
foreach(_tgt llama-mtmd-cli llama-mtmd-debug llama-llava-cli llama-gemma3-cli llama-minicpmv-cli llama-qwen2vl-cli)
    if (TARGET ${_tgt})
        set_target_properties(${_tgt} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
endforeach()

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
    src/runner_tts.cpp
    src/meta.cpp
)

# Core library
add_library(nrvna_core STATIC ${NRVNA_SOURCES})
target_include_directories(nrvna_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
)
target_link_libraries(nrvna_core PUBLIC
    llama
    mtmd
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

# nrvna-web disabled until httplib.h location resolved after llama.cpp update
# add_executable(nrvna-web cli/nrvna-web.cpp)
# target_include_directories(nrvna-web PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/server
#     ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/common
# )
# target_link_libraries(nrvna-web nrvna_core)

# Install targets
install(TARGETS nrvnad wrk flw
    RUNTIME DESTINATION bin
)
