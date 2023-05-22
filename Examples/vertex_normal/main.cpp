#include <iostream>
#include <Eigen/Dense>
#include "TriangleMesh.h" 
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include "MeshHelper.h"
#include "heartlib.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <omp.h>
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

using namespace iheartmesh;

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    igl::readOBJ(argc>1?argv[1]:DATA_PATH / "small_bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshF);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    heartlib ihla(triangle_mesh, P);
    std::vector<Eigen::Matrix<double, 3, 1>> N(meshV.rows());

    std::vector<Eigen::Matrix<double, 3, 1>> mean_curvature_normal(meshV.rows());

    std::vector<Eigen::Matrix<double, 3, 1>> gaussian_curvature_normal(meshV.rows());

    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < meshV.rows(); ++i)
    {
        // vertex normal
        Eigen::Matrix<double, 3, 1> n = ihla.VertexNormal(i);
        N[i] = n/n.norm();
        // gaussian curvature normal
        Eigen::Matrix<double, 3, 1> gaussian = ihla.KN(i);
        gaussian_curvature_normal[i] = gaussian;
        // mean
        Eigen::Matrix<double, 3, 1> mean = ihla.HN(i);
        mean_curvature_normal[i] = mean;
    } 
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("Vertex Normal", N); 
    polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("Gaussian Curvature Normal", gaussian_curvature_normal); 
    polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("Mean Curvature Normal", mean_curvature_normal); 
    polyscope::show();
    return 0;
}
