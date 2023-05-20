#include <iostream>
#include <Eigen/Dense>
#include "TriangleMesh.h" 
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include "MeshHelper.h"
#include "heartlib.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

using namespace iheartmesh;

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    igl::readOBJ(argc>1?argv[1]: DATA_PATH / "cactus.obj", meshV, meshF);
    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshF);
    // Initialize polyscope
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    heartlib ihla(triangle_mesh, P);
    std::vector<Eigen::Matrix<double, 3, 1>> N;
    std::vector<double> gaussian_curvature(meshV.rows());
    std::vector<double> mean_curvature(meshV.rows());
    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < meshV.rows(); ++i)
    {
        double gauss = ihla.K(i);
        gaussian_curvature[i] = gauss;
        // mean curvature
        double k = ihla.H(i);
        mean_curvature[i] = k;
    } 
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::getSurfaceMesh("my mesh")->addVertexScalarQuantity("gaussian curvature", gaussian_curvature);
    polyscope::getSurfaceMesh("my mesh")->addVertexScalarQuantity("mean curvature", mean_curvature);
    polyscope::show();
    return 0;
}
