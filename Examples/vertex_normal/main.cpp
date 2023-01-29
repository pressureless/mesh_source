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
    igl::readOBJ("../../../../models/small_bunny.obj", meshV, meshF);
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
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Eigen::Matrix<double, 3, 1> n = ihla.VertexNormal(i);
        N.push_back(n);
        // std::cout<<"ihla.getVertexNormal(i):\n"<<n<<std::endl;
    } 
    polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("VertexNormal", N); 
    polyscope::show();
    return 0;
}
