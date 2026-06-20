cmake_minimum_required(VERSION 3.18)
project(opencv_preview)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add executable
add_executable(opencv_preview main.cpp)

# Link OpenCV libraries
target_link_libraries(opencv_preview ${OpenCV_LIBS})

# Copy preview.png to build directory on every build
add_custom_target(copy_preview ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/../preview.png
            ${CMAKE_BINARY_DIR}/preview.png
)

# Ensure it runs before the executable
add_dependencies(opencv_preview copy_preview)


