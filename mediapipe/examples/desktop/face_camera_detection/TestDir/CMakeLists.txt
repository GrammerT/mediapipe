cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(DynamicDllLoaderTest)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加可执行文件
add_executable(DynamicDllLoaderTest main.cpp)

# 链接必要的库（如果需要）
# if(WIN32)
#     target_link_libraries(DynamicDllLoaderTest PRIVATE kernel32)
# endif()