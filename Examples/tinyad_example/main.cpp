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

void test(){
    Eigen::MatrixXi Tet(2,3);
    Tet <<
    0,1,2,
    2,1,3; 
    TriangleMesh triangle_mesh_0;
    triangle_mesh_0.initialize(Tet);

    std::vector<Eigen::Matrix<double, 2, 1>> P;
    Eigen::Matrix<double, 2, 1> P0;
    P0 << 0, 1;
    Eigen::Matrix<double, 2, 1> P1;
    P1 << 1, 0;
    Eigen::Matrix<double, 2, 1> P2;
    P2 << 1, 2;
    Eigen::Matrix<double, 2, 1> P3;
    P3 << 2, 1; 
    P.push_back(P0);
    P.push_back(P1);
    P.push_back(P2);
    P.push_back(P3);
    //
    std::vector<Eigen::Matrix<double, 2, 1>> x;
    Eigen::Matrix<double, 2, 1> x0;
    x0 << 0.2, 1;
    Eigen::Matrix<double, 2, 1> x1;
    x1 << 1.3, 0;
    Eigen::Matrix<double, 2, 1> x2;
    x2 << 1, 2.1;
    Eigen::Matrix<double, 2, 1> x3;
    x3 << 2.4, 1.1; 
    x.push_back(x0);
    x.push_back(x1);
    x.push_back(x2);
    x.push_back(x3);

    iheartmesh ihla(triangle_mesh_0, P);

    double energy = ihla.Energy(x);

    // vertex one ring
    std::cout<<"energy: "<<energy<<std::endl;
}

int main(int argc, const char * argv[]) {
    test();
    return 0;
}
