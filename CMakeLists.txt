cmake_minimum_required(VERSION 3.18)
project(MD_MLP LANGUAGES CXX CUDA)

execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE MD_BUILD_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT MD_BUILD_ID)
    set(MD_BUILD_ID "unknown")
endif()

# C++17 を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

find_package(Python3 COMPONENTS Development REQUIRED)

set(MD_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to build for, for example 86 or 80;86")
set(MD_TORCH_CUDA_ARCH_LIST "" CACHE STRING "Optional Torch CUDA architecture list, for example 8.6 or 7.5")
set(MD_CURAND_LIBRARY "" CACHE FILEPATH "Optional absolute path to libcurand")
set(LIBTORCH_PATH "" CACHE PATH "Optional path to the Python torch package or libtorch root")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Optional ONNX Runtime root. Only linked when set.")

if(LIBTORCH_PATH AND NOT Torch_DIR)
    set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch" CACHE PATH "Path to TorchConfig.cmake" FORCE)
endif()

if(NOT MD_TORCH_CUDA_ARCH_LIST)
    set(_md_torch_arches "")
    foreach(_md_arch IN LISTS MD_CUDA_ARCHITECTURES)
        if(_md_arch MATCHES "^([0-9])([0-9])([a-z]?)$")
            list(APPEND _md_torch_arches "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${CMAKE_MATCH_3}")
        else()
            list(APPEND _md_torch_arches "${_md_arch}")
        endif()
    endforeach()
    list(JOIN _md_torch_arches " " MD_TORCH_CUDA_ARCH_LIST)
endif()
set(ENV{TORCH_CUDA_ARCH_LIST} "${MD_TORCH_CUDA_ARCH_LIST}")
set(TORCH_CUDA_ARCH_LIST "${MD_TORCH_CUDA_ARCH_LIST}")

find_package(Torch REQUIRED)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

set(MD_CURAND_LINK_LIBRARY curand)
if(MD_CURAND_LIBRARY)
    set(MD_CURAND_LINK_LIBRARY "${MD_CURAND_LIBRARY}")
elseif(Torch_DIR)
    get_filename_component(_md_torch_site_packages "${Torch_DIR}/../../../.." ABSOLUTE)
    set(_md_torch_curand_versioned_library "${_md_torch_site_packages}/nvidia/curand/lib/libcurand.so.10")
    if(EXISTS "${_md_torch_curand_versioned_library}")
        set(MD_CURAND_LINK_LIBRARY "${_md_torch_curand_versioned_library}")
    else()
        find_library(_md_torch_curand_library
            NAMES curand
            HINTS "${_md_torch_site_packages}/nvidia/curand/lib"
            NO_DEFAULT_PATH
        )
        if(_md_torch_curand_library)
            set(MD_CURAND_LINK_LIBRARY "${_md_torch_curand_library}")
        endif()
    endif()
endif()

if(ONNXRUNTIME_ROOT)
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include"
        REQUIRED
    )
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib"
        REQUIRED
    )
endif()

# ソースファイルのリスト
file(GLOB_RECURSE MD_SOURCES 
    "src/md/*.cpp" 
    "src/md/*.cu"
)

# 実行ファイルを作成
add_executable(MD_MLP
    src/main.cpp
    ${MD_SOURCES}
)

target_include_directories(MD_MLP PRIVATE 
    ${PROJECT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

if(ONNXRUNTIME_ROOT)
    target_include_directories(MD_MLP PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
endif()

# libtorch をリンク
target_link_libraries(MD_MLP PRIVATE
    ${MD_CURAND_LINK_LIBRARY}
    Python3::Python
    ${TORCH_LIBRARIES} 
)

if(MD_CURAND_LINK_LIBRARY MATCHES "^/")
    get_filename_component(_md_curand_library_dir "${MD_CURAND_LINK_LIBRARY}" DIRECTORY)
    target_link_options(MD_MLP PRIVATE "-Wl,-rpath,${_md_curand_library_dir}")
endif()

if(ONNXRUNTIME_ROOT)
    target_link_libraries(MD_MLP PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()

target_compile_features(MD_MLP PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(MD_MLP PRIVATE MD_BUILD_ID="${MD_BUILD_ID}")

target_compile_options(MD_MLP PRIVATE 
    ${TORCH_CXX_FLAGS_LIST}
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-O3,-fPIC
    >
)

set_target_properties(MD_MLP PROPERTIES
    CUDA_ARCHITECTURES "${MD_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Example:
# cmake -G Ninja -S . -B build \
#   -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
#   -D Torch_DIR=/path/to/torch/share/cmake/Torch \
#   -D MD_CUDA_ARCHITECTURES=86
# ninja -C build
