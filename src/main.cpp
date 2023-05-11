//
//  main.cpp
//  DEC
//
//  Created by pressure on 10/31/22.
//

#include <iostream>
#include <Eigen/Dense>
#include "PointCloud.h" 
#include "TriangleMesh.h" 
#include "Tetrahedron.h" 
#include "PolygonMesh.h" 
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include "MeshHelper.h"
#include "dec_util.h"
#include "iheartmesh.h"

using Eigen::MatrixXi;
using Eigen::MatrixXd; 

void test_tet_mesh(){
    // MatrixXi Tet(2,4);
    // Tet <<
    // 1,2,3,0,
    // 1,2,3,4; 
    // Tetrahedron triangle_mesh_0;
    // triangle_mesh_0.initialize(Tet);

    // iheartmesh ihla(triangle_mesh_0);

    // // vertex one ring
    // std::vector<int > vertexOneRingRes = ihla.VertexOneRing(1);
    // assert(vertexOneRingRes.size() == 4);
    // assert(vertexOneRingRes[0] == 0);
    // assert(vertexOneRingRes[1] == 2);
    // assert(vertexOneRingRes[2] == 3);
    // assert(vertexOneRingRes[3] == 4);

    // // edge index
    // int eI = ihla.EdgeIndex(2, 4);
    // assert(eI == 7);

    // // face index
    // int fI = ihla.FaceIndex(1, 2, 3);
    // assert(fI == 3);

    // // face orientation
    // std::tuple< int, int > neighborT = ihla.NeighborVerticesInFace(3, 2);
    // assert(std::get<0>(neighborT) == 3);
    // assert(std::get<1>(neighborT) == 1);
}

void test_point_cloud(){
    // Eigen::MatrixXd P(4, 2);
    // // Eigen::Matrix<double, 4, 2> P;
    // P<< 0.2, 1,
    //     1, 0.4,
    //     1.3, 2,
    //     2, 1.4;  
    // std::vector<Eigen::VectorXd> PP;

    // PointCloud point_cloud(PP);

}

void test_polygon_mesh(){
    // std::vector<std::vector<int> > T;
    // std::vector<int> f1 = {0, 1, 2};
    // std::vector<int> f2 = {1, 2, 3, 4};
    // T.push_back(f1);
    // T.push_back(f2);
    // PolygonMesh polygon_mesh;
    // polygon_mesh.initialize(T);

    
}


void test_triangle_mesh(){
    //     2
    // 0       3
    //     1
    MatrixXi face(2,3);
    face <<
    0,1,2,
    2,1,3;

    std::vector<Eigen::Matrix<double, 3, 1>> P;
    Eigen::Matrix<double, 3, 1> P1;
    P1<< 0.2, 1, 0;
    Eigen::Matrix<double, 3, 1> P2;
    P2<< 1, 0.4, 0;
    Eigen::Matrix<double, 3, 1> P3;
    P3<< 1.3, 2, 0;
    Eigen::Matrix<double, 3, 1> P4;
    P4<< 2, 1.4, 0;  
    P.push_back(P1);
    P.push_back(P2);
    P.push_back(P3);
    P.push_back(P4);

    
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(face);


    iheartmesh ihla(triangle_mesh, P);

    std::tuple< int, int > rhs_4 = ihla.OrientedOppositeFaces(1, 2);
    int f_1 = std::get<0>(rhs_4);
    int f_2 = std::get<1>(rhs_4);

    std::cout<<"f_1:"<<f_1<<", f_2: "<<f_2<<std::endl;
}

int main(int argc, const char * argv[]) { 
    // from scomplex.pdf
    // test_point_cloud();
    test_triangle_mesh();
    // test_polygon_mesh();
    //
    // std::cout << "Pass!"<<std::endl;
    return 0;
}
