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
#include "dec_util.h"
#include "iheartmesh.h"

using Eigen::MatrixXi;
using Eigen::MatrixXd; 

int main(int argc, const char * argv[]) {
    MatrixXi Face(2,3);
    Face <<
    0,1,2,
    2,1,3; 
    // from scomplex.pdf
    MatrixXi Tet(2,4);
    Tet <<
    0,1,2,3,
    1,2,3,4; 
    TriangleMesh triangle_mesh_0;
    triangle_mesh_0.initialize(Tet);

    iheartmesh ihla(triangle_mesh_0);

    // vertex one ring
    std::vector<int > vertexOneRingRes = ihla.VertexOneRing(1);
    assert(vertexOneRingRes.size() == 4);
    assert(vertexOneRingRes[0] == 0);
    assert(vertexOneRingRes[1] == 2);
    assert(vertexOneRingRes[2] == 3);
    assert(vertexOneRingRes[3] == 4);

    // edge index
    int eI = ihla.EdgeIndex(2, 4);
    assert(eI == 7);

    // face index
    int fI = ihla.FaceIndex(1, 2, 3);
    assert(fI == 3);

    // face orientation
    std::tuple< int, int > neighborT = ihla.NeighborVerticesInFace(3, 2);
    assert(std::get<0>(neighborT) == 3);
    assert(std::get<1>(neighborT) == 1);

    //

    std::cout << "Pass!"<<std::endl;
    return 0;
}
