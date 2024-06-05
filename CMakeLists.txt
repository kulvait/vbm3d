cmake_minimum_required(VERSION 3.9)
project(VBM3D)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

message(STATUS "Configuring main project: " ${CMAKE_PROJECT_NAME})

set(PROJECT_VERSION_MAJOR  1)
set(PROJECT_VERSION_MINOR  0)
set(PROJECT_VERSION_PATCH  0)

set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    CACHE INTERNAL "The project version.")

###############################################################################
# Set CFLAGS and CXXFLAGS
###############################################################################
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG -w -s -ffast-math -fomit-frame-pointer -fstrength-reduce")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -w -s -ffast-math -fomit-frame-pointer -fstrength-reduce")


#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O3 -w -s -ffast-math -fomit-frame-pointer -fstrength-reduce")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g -w -s -ffast-math -fomit-frame-pointer -fstrength-reduce")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -g -w -s -ffast-math -fomit-frame-pointer -fstrength-reduce")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DDEBUG")

###############################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

###############################################################################

find_package (TIFF REQUIRED)
find_package (JPEG REQUIRED)
find_package (PNG REQUIRED)
include_directories (PUBLIC ${TIFF_INCLUDE_DIR} PUBLIC ${JPEG_INCLUDE_DIR} PUBLIC ${PNG_INCLUDE_DIRS})
link_libraries (${TIFF_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES})

###############################################################################

find_path (FFTW_INCLUDES fftw3.h)
find_library (FFTWF_LIBRARIES NAMES fftw3f)
link_libraries (${TIFF_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${FFTWF_LIBRARIES})

###############################################################################

FIND_PACKAGE( OpenMP QUIET)
if(OPENMP_FOUND)
#	message("-- OpenMP found.")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###############################################################################

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(multiscaler)
add_subdirectory(src)
add_subdirectory(src/utils)
add_subdirectory(tvl1flow)
