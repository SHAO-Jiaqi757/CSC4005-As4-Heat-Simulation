cmake_minimum_required(VERSION 3.2)
project(csc4005_as4)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode: add race condition detector")
#     set(CMAKE_CXX_FLAGS "-fsanitize=thread")
endif()
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)

include_directories(
        include
        ${MPI_CXX_INCLUDE_DIRS})


# pthread
add_executable(pthread src/pthread.cpp)
target_link_libraries(pthread ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})
target_compile_options(pthread PRIVATE -Wall -Wpedantic -DImDrawIdx=unsigned)
# omp
add_executable(omp src/omp.cpp)
target_link_libraries(omp ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_compile_options(omp PRIVATE -Wall -Wpedantic -DImDrawIdx=unsigned)


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("release mode include cuda")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/opt/rh/devtoolset-10/root/usr/bin/gcc -gencode=arch=compute_75,code=[sm_75,compute_75]")
    add_executable(cuda src/cuda.cu)
    target_link_libraries(cuda ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})
    set_target_properties(
        cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES OFF
        )
endif()