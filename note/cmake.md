cmake

不需要手动配置makefiles

cmake通过cmakelists.txt生成对饮环境下的原生的

makefile等



so c++工具链是必须的





```
cmake minimum required(VERSION 3.10)
project(Example)
```

cmake通过cmakelists.txt生成对饮环境下的原生的

makefile等

这个叫config



build



## **介绍**

CMake是一个跨平台的开源构建系统，它用于管理软件项目的构建过程。它可以生成适合各种操作系统和编译器的构建文件。CMake的编译主要有以下步骤：

1. 编写CMakeLists.txt文件
2. 用cmake命令将CMakeLists.txt文件转化为make所需要的makefile文件
3. 用make命令编译源码生成可执行文件或库

一般把CMakeLists.txt文件放在工程目录下，具体编译执行命令为：

```text
mkdir build && cd build # cmake命令指向CMakeLists.txt所在的目录，例如cmake .. 表示CMakeLists.txt在当前目录的上一级目录。cmake后会生成很多编译的中间文件以及makefile文件，所以新建的build文件夹专门用来编译
cmake .. # cmake .. 在build里生成Makefile
make # make根据生成makefile文件，编译程序，make应当在有Makefile的目录下，根据Makefile生成可执行文件。
```

## **CMakeLists.txt常用命令及流程**

编写CMakeLists.txt最常用的功能就是调用其他的.h头文件和.so/.a库文件，将.cpp/.c/.cc文件编译成可执行文件或者新的库文件。

### **CMakeLists.txt的常用命令**

- 设置project名称
  `project(xxx)`
  会自动创建两个变量，`PROJECT_SOURCE_DIR` 和 `PROJECT_NAME`

- - `${PROJECT_SOURCE_DIR}` : 本CMakeLists.txt所在的文件夹路径
  - `${PROJECT_NAME}` : 本 CMakeLists.txt的project名称

- 获取路径下的所有.cpp/.c/.cc文件，并赋值给变量中
  `aux_source_directory(路径 变量)`

- 给文件名/路径名或者其他字符串起别名，用`${变量}`获取变量内容
  `set(变量 文件名/路径/...)`

- 添加编译选项
  `add_definitions(编译选项)`

- 打印消息
  `message(消息)`

- 编译子文件夹的CMakeLists.txt
  `add_subdirectory(子文件夹名称)`

- 将.cpp/.c/.cc文件生成.a静态库
  注意，此时库文件名称通常为libxxx.so，在这里只需要写xxx即可
  `add_library(库文件名称如xxx STATIC 文件)`

- 将.cpp/.c/.cc文件生成可执行文件
  `add_executable(可执行文件名称 文件)`

- 规定.h头文件路径
  `include_directories(路径)`

- 规定.so/.a库文件路径
  `link_directories(路径)`

- 对add_library或者add_executable生成的文件进行链接操作
  注意，此时库文件名称通常为libxxx.so，在这里只需要写xxx即可
  `target_link_libraries(库文件名称/可执行文件名称 链接的库文件名称)`

### **CMakeLists.txt的基本流程**

```cmake
project(xxx) # 必须

add_subdirectory(子文件夹名称) # 父目录必须，子目录没有下级子目录则不需要

add_library(库文件名称 STATIC 文件) # 通常子目录（二选一）
add_executable(可执行文件名称 文件)  # 通常父目录（二选一）

include_directories(路径) # 必须
link_directories(路径) # 必须

target_link_libraries(库文件名称/可执行文件名称 链接的库文件名称) # 必须
```

## **具体编写步骤**

### **1. 声明的cmake最低版本**

```cmake
cmake_minimum_required( VERSION 3.4 )
```

### **2. 检查C++版本，添加c++标准支持（Optional）**

```cmake
# 添加c++11标准支持 【可选】
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 检查C++版本 【可选】 ， Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
```

### **3. 添加工程名称（可任取）**

会自动创建两个变量，`PROJECT_SOURCE_DIR` 和 `PROJECT_NAME`

- `${PROJECT_SOURCE_DIR}` : 本CMakeLists.txt所在的文件夹路径
- `${PROJECT_NAME}` : 本 CMakeLists.txt的project名称

```cmake
PROJECT(TEST)
MESSAGE(STATUS "Project: SERVER") #打印相关消息消息
```

### **4. 设置编译模式**

```cmake
# 设置为 Release 模式
SET(CMAKE_BUILD_TYPE Release)

# 或者，设置为 debug 模式
SET(CMAKE_BUILD_TYPE debug)

# 打印设置的编译模型信息
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

```

### **4. 添加子目录**

如果项目包含多个子模块或子目录，可以使用 `add_subdirectory()` 指令将它们添加到构建过程中。

```cmake
add_subdirectory(submodule_dir)
```

### **5. 添加头文件**

- 举例使用OpenCV库
  备注：这里的OpenCV包含目录为含有OpenCVConfig.cmake的路径。
  set(OpenCV_DIR "/usr/local/include/opencv3.2.0/share/OpenCV")
  find_package(OpenCV REQUIRED)
  include_directories( \${OpenCV_INCLUDE_DIRS} )
- 如果需要添加所有包含的.h头文件
  include_directories(
  \${PROJECT_SOURCE_DIR}/../include/dir1
  \${PROJECT_SOURCE_DIR}/../include/dir2
  )
- 包含第三库的头文件，举例第三方库的名字为LIBa
  \#设置.h文件对应的路径
  set( LIBa_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ThirdParty/LIBa/include/)
  
  \#包含.h文件路径
  include_directories( \${OpenCV_INCLUDE_DIRS}
  \${LIBa_INCLUDE_DIRS}
  \${LIBa_INCLUDE_DIRS}/LIBa/)
  包含第三方库的cpp文件
  set(LIBa_SRCS "${PROJECT_SOURCE_DIR}/ThirdParty/LIBa/src")

### **6. 添加源代码路径**

通过设定SRC变量，将源代码路径都给SRC，如果有多个，可以直接在后面继续添加

```cmake
set(SRC 
    ${PROJECT_SOURCE_DIR}/../include/dir1/func1.cpp 
    ${PROJECT_SOURCE_DIR}/../include/dir2/func2.cpp 
    ${PROJECT_SOURCE_DIR}/main.cpp 
    )
```

### **7. 创建共享库/静态库**

设置生成共享库的路径

```cmake
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# 即生成的共享库在工程文件夹下的lib文件夹中
```

创建共享库（把工程内的cpp文件都创建成共享库文件，方便通过头文件来调用）。这时候只需要cpp，不需要有主函数

```cmake
set(LIB_NAME main_lib)
# ${LIB_NAME}是生成的库的名称 表示生成的共享库文件就叫做 lib工程名.so
# 也可以专门写cmakelists来编译一个没有主函数的程序来生成共享库，供其它程序使用
add_library(${LIB_NAME} STATIC ${SRC}) # SHARED为生成动态库，STATIC为生成静态库
```

### **8. 链接库文件**

把刚刚生成的${LIB_NAME}库和所需的其它库链接起来

如果需要链接其他的动态库，-l后面届 去除lib前缀和.so后缀的名称（即为LIB_NAME），以链接

以 libpthread.so 为例, -lpthread

```cmake
target_link_libraries(${LIB_NAME} pthread dl)
```

### **9. 编译主函数，生成可执行文件**

先设置路径

```cmake
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
```

可执行文件生成`add_executable( 目标文件（可执行文件） 依赖文件（.cpp）)`

```cmake
add_executable(${PROJECT_NAME} ${SRC})
```

这个可执行文件所需的库（一般就是刚刚生成的工程的库咯）

```cmake
target_link_libraries(${PROJECT_NAME} pthread dl ${LIB_NAME})
```

## **CMakeLists.txt例子**

```cmake
cmake_minimum_required( VERSION 2.8 )
project( loop_closure )

#set(CMAKE_BUILD_TYPE  Debug)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# 使用opencv库
# set(OpenCV_DIR "/usr/local/include/opencv3.2.0/share/OpenCV")
set(OpenCV_DIR "/opt/ros/kinetic/share/OpenCV-3.3.1-dev")
find_package(OpenCV REQUIRED)


set( DBoW3_INCLUDE_DIRS "/usr/local/include")

set( DBoW2_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/ThirdParty/DBow-master/include/)
message(${DBoW2_INCLUDE_DIRS})

#important
#file(GLOB DBoW2_SRCS ${PROJECT_SOURCE_DIR}/ThirdParty/DBow-master/src/*.cpp)
#message(${DBoW2_SRCS})

set(DBoW2_SRCS "${PROJECT_SOURCE_DIR}/ThirdParty/DBow-master/src")
message(${DBoW2_SRCS})

find_package(DLib QUIET
             PATHS ${DEPENDENCY_INSTALL_DIR})
if(${DLib_FOUND})
   message("DLib library fo {DEPENDENCY_DIR}
            GIT_REPOSITORY http://github.com/dorian3d/DLib
            GIT_TAG master
      INSTALL_DIR ${DEPENDENCY_INSTALL_DIR}
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
   if(${DOWNLOAD_DLib_dependency})
      add_custom_target(Dependencies ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR} DEPENDS DLib)
   else()
      message(SEND_ERROR "Please, activate DOWNLOAD_DLib_dependency option or download manually")
   endif(${DOWNLOAD_DLib_dependency})
endif(${DLib_FOUND})


include_directories( ${OpenCV_INCLUDE_DIRS}
                     ${DBoW3_INCLUDE_DIRS} 
                     ${DBoW2_INCLUDE_DIRS} 
                     ${DBoW2_INCLUDE_DIRS}/DBoW2/)

message("DBoW3_INCLUDE_DIRS ${DBoW3_INCLUDE_DIRS}")
message("DBoW2_INCLUDE_DIRS ${DBoW2_INCLUDE_DIRS}")
message("opencv ${OpenCV_VERSION}")

# dbow3 is a simple lib so I assume you installed it in default directory

set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a")

add_executable(${PROJECT_NAME} src/loop_closure.cpp  src/run_main.cpp
               ${DBoW2_SRCS}/BowVector.cpp ${DBoW2_SRCS}/FBrief.cpp 
               ${DBoW2_SRCS}/FeatureVector.cpp ${DBoW2_SRCS}/FORB.cpp 
               ${DBoW2_SRCS}/FSurf64.cpp ${DBoW2_SRCS}/QueryResults.cpp ${DBoW2_SRCS}/ScoringObject.cpp)

message(${DBoW2_SRCS}/BowVector.cpp)

target_link_libraries(${PROJECT_NAME} 
                      ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW3_LIBS})
```

