//
//  main.cpp
//  DEC
//
//  Created by pressure on 10/31/22.
//

#include <iostream>
#include <Eigen/Dense>
#include "TriangleMesh.h" 
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include "MeshHelper.h"
#include "iheartmesh.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <omp.h>
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    igl::readOBJ(DATA_PATH / "small_bunny.obj", meshV, meshF);
    // igl::readOFF("../../../models/bunny_1k.off", meshV, meshF); 
    // igl::readOFF("../../../models/bunny_200.off", meshV, meshF); 
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshF);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    iheartmesh ihla(triangle_mesh, P);
    std::vector<Eigen::Matrix<double, 3, 1>> N(meshV.rows());

    std::vector<Eigen::Matrix<double, 3, 1>> mean_curvature_normal(meshV.rows());

    std::vector<Eigen::Matrix<double, 3, 1>> gaussian_curvature_normal(meshV.rows());

    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < meshV.rows(); ++i)
    {
        // vertex normal
        Eigen::Matrix<double, 3, 1> n = ihla.VertexNormal(i);
        N[i] = n;
        // gaussian curvature normal
        Eigen::Matrix<double, 3, 1> gaussian = ihla.KN(i);
        gaussian_curvature_normal[i] = gaussian;
        // printf("Thread %d is doing iteration %d.\n", omp_get_thread_num(), i);
        // mean
        Eigen::Matrix<double, 3, 1> mean = ihla.HN(i);
        mean_curvature_normal[i] = mean;
        // std::cout<<"ihla.getVertexNormal(i):\n"<<n<<std::endl;
    } 
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("Vertex Normal", N); 
    polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("Gaussian Curvature Normal", gaussian_curvature_normal); 
    polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("Mean Curvature Normal", mean_curvature_normal); 
    polyscope::show();
    return 0;
}
