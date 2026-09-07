cmake_minimum_required(VERSION 3.16)
project(dfuzz CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -ffast-math must never be enabled: every bound here assumes IEEE semantics.
# buildinfo.cpp reports the flag's state at runtime so a violation is visible
# in the result record rather than silent.
add_compile_options(-Wall -Wextra -fno-fast-math)

# ggml, built CPU-only. Metal and Accelerate are OFF deliberately: Accelerate
# can reroute kernels through BLAS, which would change what is being measured.
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "")
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "")
set(LLAMA_CURL           OFF CACHE BOOL "")
set(GGML_METAL           OFF CACHE BOOL "")
set(GGML_ACCELERATE      OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "")
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

add_library(dfuzz_core STATIC
  src/reference.cpp src/bound.cpp src/formats.cpp
  src/buildinfo.cpp src/init.cpp src/oracle.cpp
  src/shrink.cpp src/generate.cpp src/gemv.cpp)
target_include_directories(dfuzz_core PUBLIC include)
target_link_libraries(dfuzz_core PUBLIC ggml-cpu ggml-base)

add_executable(diag_gemm tools/diag_gemm.cpp)
target_link_libraries(diag_gemm PRIVATE dfuzz_core)

add_executable(dfuzz_sweep src/sweep.cpp)
target_link_libraries(dfuzz_sweep PRIVATE dfuzz_core)

enable_testing()
foreach(t smoke smoke_fp4 dequant_check dequant_fp4 shrink_test bound_sensitivity gemv_test gemm_test bench_oracle)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE dfuzz_core)
  if(NOT ${t} STREQUAL bench_oracle)
    add_test(NAME ${t} COMMAND ${t})
  endif()
endforeach()
