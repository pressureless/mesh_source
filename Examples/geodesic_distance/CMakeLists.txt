cmake_minimum_required(VERSION 3.19.0)

project("Vertex normal")

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
    GIT_TAG 3.3
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

# OpenMP
find_package(OpenMP)

# Create an executable
set(SRCS 
  main.cpp
  ../../src/Connectivity.cpp
  ../../src/dec_util.cpp
  ../../src/MeshHelper.cpp
  ../../src/PointCloud.cpp
  ../../src/PolygonMesh.cpp
  ../../src/Tetrahedron.cpp
  ../../src/TriangleMesh.cpp
)

add_executable(main "${SRCS}")

# Include settings
target_include_directories(main PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../include/")

# Link settings
target_link_libraries(main igl::core polyscope Eigen3::Eigen autodiff::autodiff OpenMP::OpenMP_CXX)
