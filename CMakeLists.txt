cmake_minimum_required(VERSION 2.8)
project(stapleCC)
aux_source_directory(./src/ SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

IF (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3") #c++11
ENDIF (UNIX)
SET(OpenCV_DIR D:\\project\\opencv\\build)
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# python wrapper
add_library(Staple SHARED ./staple_wrapper.cpp ${SRC_LIST})
include_directories(src)

# dirty python dev
SET(Python3_INCLUDE_DIRS "D:\\software\\Anaconda3\\include")
SET(Python3_LIBRARIES "D:\\software\\Anaconda3\\libs\\python36.lib")
include_directories(${Python3_INCLUDE_DIRS})

find_package(Boost 1.66.0 COMPONENTS python numpy REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
message(${Boost_LIBRARIES})
target_link_libraries(Staple ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Python3_LIBRARIES})
