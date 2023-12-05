cmake_minimum_required(VERSION 3.8)
project(final_project VERSION 0.1.0 LANGUAGES CXX CUDA)

# 包含头文件目录
include_directories(include)
include_directories(include/cuda)
include_directories(include/cpp)

# 指定源文件
file(GLOB CPP_SOURCES "src/*.cpp" "src/cpp/*.cpp")
file(GLOB CUDA_SOURCES "cuda/*.cu")

# 创建可执行文件
add_executable(final_project ${CPP_SOURCES} ${CUDA_SOURCES})

set_target_properties(final_project PROPERTIES CUDA_ARCHITECTURES "75")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
