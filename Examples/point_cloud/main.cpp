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
#include "polyscope/point_cloud.h"
#include "PointCloud.h"
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    igl::readOBJ(argc>1?argv[1]:DATA_PATH / "small_bunny.obj", meshV, meshF);
    // igl::readOBJ("../../../models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    // TriangleMesh triangle_mesh;
    // triangle_mesh.initialize(meshF);
    // Initialize polyscope
    std::vector<Eigen::VectorXd> PN;
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        PN.push_back(meshV.row(i).transpose());
        P.push_back(meshV.row(i).transpose());
    }

    std::vector<std::vector<size_t>> neighbors = GetPointNeighbors(PN, 6);

    PointCloud pc(PN, neighbors);
    iheartmesh ihla(pc, P);
    std::vector<Eigen::Matrix<double, 3, 3>> cov;

    // std::cout<<"v:"<<meshV.rows()<<std::endl;

    std::vector<Eigen::VectorXd> point_normals(meshV.rows());

    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Eigen::VectorXd n = ihla.Normal(i);

        // Naive way to check whether we need to flip the normal (for convex .obj)
        // if (P[i].dot(n) < 0)
        // {
        //     n = -n;
        // }
        point_normals[i] = n;
    } 
    polyscope::init();   
    polyscope::PointCloud* psCloud = polyscope::registerPointCloud("really great points", P);

    // set some options
    psCloud->setPointRadius(0.01);
    psCloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
    psCloud->addVectorQuantity("Normals", point_normals);

    // show
    polyscope::show();
    return 0;
}
