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
add_library(Staple SHARED ./staple_wrapper.cpp)
include_directories(src)

# dirty python dev
SET(Python3_INCLUDE_DIRS "D:\\software\\Anaconda3\\envs\\py35\\include")
SET(Python3_LIBRARIES "D:\\software\\Anaconda3\\envs\\py35\\libs\\python35.lib")
include_directories(${Python3_INCLUDE_DIRS})

SET(Boost_DIR D:\\software\\boost_1_65_1)
find_package(Boost 1.65.1 COMPONENTS python REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
message(${Boost_LIBRARIES})
target_link_libraries(Staple ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Python3_LIBRARIES})
