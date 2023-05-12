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
#include "MeshHelper.h"
#include "iheartmesh.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <experimental/filesystem> 
namespace fs = std::experimental::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);


int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("../../../models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("../../../models/cactus.obj", meshV, meshF);
    igl::readOBJ(DATA_PATH / "yog.obj", meshV, meshF);
    // igl::readOBJ("../../../models/cartoon-elephant.obj", meshV, meshF);
    // igl::readOBJ("../../../models/keenan-ogre.obj", meshV, meshF);
    // igl::readOBJ("../../../models/elephant.obj", meshV, meshF);
    // std::cout<<meshV<<std::endl;
    // std::cout<<meshF<<std::endl;
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshF);
    // Initialize polyscope
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    iheartmesh ihla(triangle_mesh, P);
    std::vector<Eigen::Matrix<double, 3, 1>> N;
    std::vector<double> gaussian_curvature(meshV.rows());
    std::vector<double> mean_curvature(meshV.rows());
    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < meshV.rows(); ++i)
    {
        // Eigen::Matrix<double, 3, 1> n = ihla.VertexNormal(i);
        // N.push_back(n);
        double gauss = ihla.K(i);
        gaussian_curvature[i] = gauss;
        // std::cout<<"i:"<<i<<", gauss: "<<gauss<<std::endl;  
        // mean curvature
        double k = ihla.H(i);
        mean_curvature[i] = k;
        // std::cout<<"i:"<<i<<", k: "<<k<<std::endl;  
    } 
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::getSurfaceMesh("my mesh")->addVertexScalarQuantity("gaussian curvature", gaussian_curvature);
    polyscope::getSurfaceMesh("my mesh")->addVertexScalarQuantity("mean curvature", mean_curvature);
    polyscope::show();
    return 0;
}
