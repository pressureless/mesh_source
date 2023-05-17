cmake_minimum_required(VERSION 3.19.0)

project("iheartmesh examples")
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
# autodiff
FetchContent_Declare(
    autodiff
    GIT_REPOSITORY https://github.com/autodiff/autodiff
    GIT_TAG v1.0.1
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
set(AUTODIFF_BUILD_TESTS OFF CACHE INTERNAL "")
set(AUTODIFF_BUILD_PYTHON OFF CACHE INTERNAL "")
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(AUTODIFF_BUILD_DOCS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable( autodiff )

# eigen
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE INTERNAL "")
set(EIGEN_SPLIT_LARGE_TESTS OFF CACHE INTERNAL "")
set(EIGEN_BUILD_BTL OFF CACHE INTERNAL "")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE INTERNAL "")
set(EIGEN_BUILD_TESTING OFF CACHE INTERNAL "")
set(BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable( Eigen3 )

# libigl
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl
    GIT_TAG v2.4.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable( libigl )

# polyscope
FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope
    GIT_TAG v1.3.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable( polyscope )

# OpenMesh
set(OPENMESH_DOCS false CACHE BOOL "Enable or disable building of documentation")
FetchContent_Declare(openmesh
    GIT_REPOSITORY https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
    GIT_TAG        OpenMesh-8.1
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(openmesh)
add_definitions(-DNO_DECREMENT_DEPRECATED_WARNINGS)

# GLFW and glad
if (NOT TARGET glfw)
    FetchContent_Declare(
        glfw
        # URL https://github.com/glfw/glfw/releases/download/3.3.8/glfw-3.3.8.zip
        # URL_HASH SHA256=4d025083cc4a3dd1f91ab9b9ba4f5807193823e565a5bcf4be202669d9911ea6
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3-stable
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable( glfw )
endif()

if (NOT TARGET glad)
    FetchContent_Declare(
            glad
            GIT_REPOSITORY https://github.com/Dav1dde/glad.git
            GIT_TAG v2.0.4
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
    )
    set(GLAD_PROFILE "core" CACHE STRING "OpenGL profile")
    set(GLAD_API "gl=" CACHE STRING "API type/version pairs, like \"gl=3.2,gles=\", no version means latest")
    set(GLAD_GENERATOR "c" CACHE STRING "Language to generate the binding for")
    add_subdirectory(${glad_SOURCE_DIR} ${glad_BINARY_DIR})
    FetchContent_MakeAvailable( glad )
endif()

# OpenMP
find_package(OpenMP)

# Add src/header to the lib
file(GLOB_RECURSE MESH_EXAMPLES_HEADERS include/*.hpp include/*.h)
file(GLOB_RECURSE MESH_EXAMPLES_SOURCES src/*.cpp)
add_library(Mesh-Examples ${MESH_EXAMPLES_SOURCES} ${MESH_EXAMPLES_HEADERS})

# Set models path
target_compile_definitions(Mesh-Examples PUBLIC DATA_PATH_STR="${CMAKE_CURRENT_SOURCE_DIR}/models")

# Include settings
target_include_directories(Mesh-Examples PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

# Link settings
target_link_libraries(Mesh-Examples igl::core polyscope Eigen3::Eigen OpenMP::OpenMP_CXX autodiff::autodiff)

file(GLOB EXAMPLE_APPS examples/*/main.cpp)
foreach(main ${EXAMPLE_APPS})
    # message(STATUS "Main is ${main}")
    get_filename_component(app_name ${main} NAME_WE)
    get_filename_component(dir ${main} DIRECTORY)
    get_filename_component(dir_name ${dir} NAME)

    # message(STATUS "Adding app_name ${app_name}")
    # message(STATUS "Adding dir ${dir}")
    # message(STATUS "Adding dir_name ${dir_name}")
    add_executable(${dir_name} ${main})
    target_include_directories(${dir_name} PUBLIC "${dir}")
    add_dependencies(${dir_name} Mesh-Examples)
    if (${dir_name} STREQUAL "polygon_mesh")
        target_link_libraries(${dir_name} PUBLIC Mesh-Examples OpenMeshCore)
    elseif (${dir_name} STREQUAL "winding_number")
        target_include_directories(${dir_name} PUBLIC "${polyscope_SOURCE_DIR}/deps/glad/include")
        target_link_libraries(${dir_name} PUBLIC Mesh-Examples OpenMeshCore glad)
    else()
        target_link_libraries(${dir_name} PUBLIC Mesh-Examples)
    endif()
endforeach()

