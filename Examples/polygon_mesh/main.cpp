#include <iostream>
#include <Eigen/Dense>
#include "TriangleMesh.h" 
#include "FaceMesh.h" 
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include "MeshHelper.h"
#include "heartlib.h"
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

using namespace iheartmesh;

typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    igl::readOBJ(argc>1?argv[1]: DATA_PATH /  "polygon/suzanne.obj", meshV, meshF);

    MyMesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, argc>1?argv[1]: DATA_PATH /  "polygon/suzanne.obj")){
        std::cout<<"file i/o error"<<std::endl;
    }
    std::vector<std::vector<int> > faces;
    for (MyMesh::FaceIter f_it=mesh.faces_begin(); f_it!=mesh.faces_end(); ++f_it) {
        std::vector<int> f_list;
        for (OpenMesh::PolyConnectivity::FaceVertexCCWIter v_iter = mesh.fv_ccwiter (*f_it);
            v_iter.is_valid(); ++v_iter)
        {
            f_list.push_back(v_iter->idx());
        }
        faces.push_back(f_list);
    }
    // Initialize triangle mesh
    PolygonMesh polygon_mesh;
    polygon_mesh.initialize(faces);
    FaceMesh face_mesh(polygon_mesh.bm1, polygon_mesh.bm2);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, faces);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    heartlib ihla(face_mesh, P);
    std::vector<Eigen::Matrix<double, 3, 1>> N;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Eigen::Matrix<double, 3, 1> n = ihla.VertexNormal(i);
        N.push_back(n/n.norm());
    } 
    polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("VertexNormal", N); 
    polyscope::show();
    return 0;
}
