cmake_minimum_required(VERSION 3.2)
project(csc4005_as4)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode: add race condition detector")
    set(CMAKE_CXX_FLAGS "-fsanitize=thread")
endif()
find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

include_directories(
        include
        imgui
        imgui/backends
        ${SDL2_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS})

file(GLOB IMGUI_SRC
        imgui/*.cpp
        imgui/backends/imgui_impl_sdl.cpp
        imgui/backends/imgui_impl_opengl2.cpp
        imgui/misc/freetype/imgui_freetype.cpp
        imgui/misc/cpp/imgui_stdlib.cpp
        )
add_library(core STATIC ${IMGUI_SRC})
# file(GLOB CSC4005_PROJECT_SRC src/*.cpp src/*.c)

get_filename_component(FONT_PATH imgui/misc/fonts/DroidSans.ttf ABSOLUTE)
target_link_libraries(core PUBLIC
        Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})
target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)

add_executable(sequential_gui src/sequential_gui.cpp src/graphic.cpp)

#sequential_gui
target_link_libraries(sequential_gui core)
target_compile_definitions(sequential_gui PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
# pthread_gui
add_executable(pthread_gui src/pthread_gui.cpp src/graphic.cpp)
target_link_libraries(pthread_gui core)
target_compile_definitions(pthread_gui PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
# omp_gui
add_executable(omp_gui src/omp_gui.cpp src/graphic.cpp)
target_link_libraries(omp_gui core)
target_compile_definitions(omp_gui PRIVATE -DFONT_PATH=\"${FONT_PATH}\")


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("release mode include cuda")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/opt/rh/devtoolset-10/root/usr/bin/gcc -gencode=arch=compute_75,code=[sm_75,compute_75]")
    add_executable(cuda_gui src/graphic.cpp src/cuda_gui.cu)
    target_link_libraries(cuda_gui core)
    set_target_properties(
        cuda_gui
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES OFF
        )
endif()
if (WIN32)
    target_link_options(sequential_gui BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
endif()