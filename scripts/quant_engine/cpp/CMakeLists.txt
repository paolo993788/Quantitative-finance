# Standalone build of the header-only C++ engines: unit tests.
#
# The Python extension is built by setup.py (pybind11); this project compiles the same headers without Python,
# with strict warnings and optional sanitizers, so that the C++ code is checked on its own.
#
#   cmake -S scripts/quant_engine/cpp -B build/cpp -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
#   cmake --build build/cpp
#   ctest --test-dir build/cpp --output-on-failure
#
# Sanitizers: -DQE_SANITIZE=address,undefined (memory errors and undefined behaviour) or -DQE_SANITIZE=thread
# (data races in the parallel loops). They cannot be combined in one build.

cmake_minimum_required(VERSION 3.16)
project(quant_engine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QE_WERROR "Treat compiler warnings as errors" ON)
set(QE_SANITIZE "" CACHE STRING "Comma-separated sanitizers passed to -fsanitize (for example address,undefined)")

find_package(Threads REQUIRED)

add_library(qe_engine INTERFACE)
target_include_directories(qe_engine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qe_engine INTERFACE Threads::Threads)

if(MSVC)
  target_compile_options(qe_engine INTERFACE /W4 /permissive- $<$<BOOL:${QE_WERROR}>:/WX>)
else()
  target_compile_options(qe_engine INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
                         -Wold-style-cast -Wnull-dereference -Wdouble-promotion
                         $<$<BOOL:${QE_WERROR}>:-Werror>)
  # Bounds-checked standard containers in debug-like builds.
  target_compile_definitions(qe_engine INTERFACE $<$<NOT:$<CONFIG:Release>>:_GLIBCXX_ASSERTIONS>)
endif()

if(QE_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "QE_SANITIZE is supported with GCC and Clang only")
  endif()
  target_compile_options(qe_engine INTERFACE -fsanitize=${QE_SANITIZE} -fno-omit-frame-pointer
                         -fno-sanitize-recover=all)
  target_link_options(qe_engine INTERFACE -fsanitize=${QE_SANITIZE})
endif()

enable_testing()

add_executable(core_tests tests/test_core.cpp)
target_link_libraries(core_tests PRIVATE qe_engine)
add_test(NAME core_tests COMMAND core_tests)

