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
#include "PolygonMesh.h"

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    igl::readOBJ(argc>1?argv[1]: DATA_PATH /  "polygon/suzanne.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);

    MyMesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, DATA_PATH / "polygon/suzanne.obj")){
        std::cout<<"file i/o error"<<std::endl;
    }
    std::vector<std::vector<int> > faces;
    for (MyMesh::FaceIter f_it=mesh.faces_begin(); f_it!=mesh.faces_end(); ++f_it) {
        std::vector<int> f_list;
        // std::cout<<"cur_face: ";
        for (OpenMesh::PolyConnectivity::FaceVertexCCWIter v_iter = mesh.fv_ccwiter (*f_it);
            v_iter.is_valid(); ++v_iter)
        {
            // std::cout<<v_iter->idx()<<", ";
            f_list.push_back(v_iter->idx());
        }
        // std::cout<<std::endl;
        faces.push_back(f_list);
    }
    // std::cout<<"Faces:"<<faces.size()<<std::endl;
    // Initialize triangle mesh
    PolygonMesh polygon_mesh;
    polygon_mesh.initialize(faces);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, faces);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    iheartmesh ihla(polygon_mesh, P);
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
