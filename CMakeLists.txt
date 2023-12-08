cmake_minimum_required(VERSION 3.8)
project(final_project VERSION 0.1.0 LANGUAGES CXX CUDA)

# 包含目录
include_directories(include)
include_directories(include/cuda)
include_directories(include/cpp)

# 指定源文件
file(GLOB CPP_SOURCES "src/*.cpp" "src/cpp/*.cpp")
file(GLOB CUDA_SOURCES "cuda/*.cu")

# 创建可执行文件
add_executable(final_project ${CPP_SOURCES} ${CUDA_SOURCES})
target_compile_options(final_project PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -dlcm=ca>)
# 查找并链接 OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(final_project PUBLIC OpenMP::OpenMP_CXX)
endif()
find_package(CUDA REQUIRED)
target_link_libraries(final_project PUBLIC ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY})
# 设置 CUDA 架构
set_target_properties(final_project PROPERTIES CUDA_ARCHITECTURES "75")

# 设置编译选项（例如，对于 Debug 模式）
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
# 查找并链接其他库（例如，Boost）
# find_package(Boost REQUIRED)
# if(Boost_FOUND)
#     include_directories(${Boost_INCLUDE_DIRS})
#     target_link_libraries(final_project ${Boost_LIBRARIES})
# endif()

# 设置项目打包信息
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
