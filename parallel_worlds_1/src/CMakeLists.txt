cmake_minimum_required( VERSION 3.12 )
project( parallel_worlds_1 )

set( EXECUTABLE ${PROJECT_NAME} )

if( MSVC )
	message( "MSVC: adding compiler flags" )
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Z7 /arch:SSE2 /I.")
endif( MSVC )

if( UNIX )
	message( "UNIX-like system: adding compiler flags" )
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -msse2 -I.")
endif( UNIX )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# THX: https://stackoverflow.com/questions/67641589/how-can-i-tell-cmake-3-9-i-want-to-use-nvidias-opencl-library
find_package( CUDA )
set(OpenCL_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
find_package( OpenCL REQUIRED )

include_directories( ${OpenCL_INCLUDE_DIRS} )
set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} CACHE FILEPATH "OpenCL include directory")
set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} CACHE FILEPATH "OpenCL libraries")

message( "OpenCL include directory is: " ${OpenCL_INCLUDE_DIRS} )

add_executable( ${EXECUTABLE} main.cpp )
target_link_libraries( ${EXECUTABLE} ${OpenCL_LIBRARIES} )
target_link_libraries( ${EXECUTABLE} ${OpenCV_LIBS} )

# copy src for .cl files so that they can probably be read if someone
# just starts the program
add_custom_command(TARGET ${EXECUTABLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}" "$<TARGET_FILE_DIR:${EXECUTABLE}>/src")
add_custom_command(TARGET ${EXECUTABLE} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/../preview.png" "$<TARGET_FILE_DIR:${EXECUTABLE}>")

