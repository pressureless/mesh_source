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


int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    igl::readOBJ("../../../models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshF);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    iheartmesh ihla(triangle_mesh, P);
    std::vector<Eigen::Matrix<double, 3, 1>> N;
    std::vector<double> gaussian_curvature;
    std::vector<double> mean_curvature;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        // Eigen::Matrix<double, 3, 1> n = ihla.VertexNormal(i);
        // N.push_back(n);
        double gauss = ihla.K(i);
        gaussian_curvature.push_back(gauss);
        // mean curvature
        double k = ihla.H(i);
        mean_curvature.push_back(k);
        // std::cout<<"i:"<<i<<", k: "<<k<<std::endl;  
    } 
    // polyscope::getSurfaceMesh("my mesh")->addVertexDistanceQuantity("GaussianCurvature", gaussian_curvature); 

    polyscope::getSurfaceMesh("my mesh")->addVertexScalarQuantity("gaussian curvature", gaussian_curvature);
    polyscope::getSurfaceMesh("my mesh")->addVertexScalarQuantity("mean curvature", mean_curvature);
    polyscope::show();
    return 0;
}
