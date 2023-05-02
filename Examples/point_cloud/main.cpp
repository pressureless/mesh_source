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

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    igl::readOBJ("../../../models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    // TriangleMesh triangle_mesh;
    // triangle_mesh.initialize(meshF);
    // Initialize polyscope
    polyscope::init();   
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
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Eigen::Matrix<double, 3, 3> c = ihla.cov(i);
        Eigen::EigenSolver<Eigen::MatrixXd> es(c);
        double sigma = ihla.Normal(i);
        // std::cout<<"eigenvalues:\n"<<es.eigenvalues()<<std::endl;
        // std::cout<<"eigenvectors:\n"<<es.eigenvectors().col(2)<<std::endl;
        // cov.push_back(c);
        std::cout<<"sigma:\n"<<sigma<<std::endl;
    } 
    polyscope::PointCloud* psCloud = polyscope::registerPointCloud("really great points", P);

    // set some options
    psCloud->setPointRadius(0.01);
    psCloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);

    // show
    polyscope::show();
    return 0;
}
