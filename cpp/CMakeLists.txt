cmake_minimum_required(VERSION 3.20)
project(rdd_detect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenVINO_DIR: the cmake folder of the openvino pip package, OpenCV_DIR: the build folder of OpenCV
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

add_executable(rdd_detect main.cpp)
target_link_libraries(rdd_detect PRIVATE openvino::runtime ${OpenCV_LIBS})
