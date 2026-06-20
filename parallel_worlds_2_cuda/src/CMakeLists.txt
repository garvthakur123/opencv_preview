cmake_minimum_required(VERSION 3.18)

# Optional: explicitly set clang++  -- g++ does not work w/ OpenCV and CUDA
if(NOT DEFINED ENV{CLANG_CXX})
  set(CMAKE_CXX_COMPILER /usr/bin/clang++)
else()
  set(CMAKE_CXX_COMPILER $ENV{CLANG_CXX})
endif()


#PFF20260321:
set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
#:PFF20260321

project(parallel_worlds_cuda LANGUAGES CXX)

# sources (search for .cpp and .cu)
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/*.cpp"
  "${CMAKE_SOURCE_DIR}/*.cu"
)

configure_file(
    ${CMAKE_SOURCE_DIR}/../preview.png
    ${CMAKE_BINARY_DIR}/preview.png
    COPYONLY
)

# handle .cu als C++ source so that it is processed by
# clang instead of nvcc/g++
foreach(f ${SRC_FILES})
  if(f MATCHES "\\.cu$")
    set_source_files_properties(${f} PROPERTIES LANGUAGE CXX)
  endif()
endforeach()

add_executable(parallel_worlds
    main.cu
    GreyFilter.cu
    SobelFilter.cu
)

# search CUDA headers and cudart (runtime)
find_path(CUDA_INCLUDE_DIR cuda.h
  PATHS /usr/include /usr/local/cuda/include /usr/local/include
)

target_include_directories(parallel_worlds SYSTEM PUBLIC ${CMAKE_SOURCE_DIR})

find_library(CUDART_LIBRARY NAMES cudart libcudart
  PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/lib64
)

if(NOT CUDA_INCLUDE_DIR OR NOT CUDART_LIBRARY)
  message(FATAL_ERROR "CUDA headers/libs not found. Set CUDA include/lib paths or install CUDA toolkit.")
endif()


include_directories(${CUDA_INCLUDE_DIR})
#link_libraries(${CUDART_LIBRARY})

# clang++ w/ CUDA needs libstdc++ (as opposed to libc++)
target_compile_options(parallel_worlds PRIVATE "-stdlib=libstdc++")

# RPATH / library path
get_filename_component(CUDART_DIR ${CUDART_LIBRARY} DIRECTORY)
#target_link_directories(parallel_worlds PRIVATE ${CUDART_DIR})
target_link_options(parallel_worlds PRIVATE "-Wl,-rpath,${CUDART_DIR}")


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75)

set_target_properties(parallel_worlds PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CXX
)

target_link_libraries(parallel_worlds
    ${OpenCV_LIBS}
    ${CUDART_LIBRARY}
)


# Debugging output
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA include: ${CUDA_INCLUDE_DIR}")
message(STATUS "CUDART lib: ${CUDART_LIBRARY}")
