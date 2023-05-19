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
#include "polyscope/point_cloud.h"
#include "PointCloud.h"
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

using namespace iheartmesh;

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    igl::readOBJ(argc>1?argv[1]:DATA_PATH / "small_bunny.obj", meshV, meshF);
    std::vector<Eigen::VectorXd> PN;
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        PN.push_back(meshV.row(i).transpose());
        P.push_back(meshV.row(i).transpose());
    }
    std::vector<std::vector<size_t>> neighbors = GetPointNeighbors(PN, 6);
    PointCloud pc(PN, neighbors);
    heartlib ihla(pc, P);
    std::vector<Eigen::VectorXd> original_normals(meshV.rows());
    std::vector<Eigen::VectorXd> point_normals(meshV.rows());
    std::vector<Eigen::VectorXd> inverse_original_normals(meshV.rows());
    std::vector<Eigen::VectorXd> inverse_point_normals(meshV.rows());

    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Eigen::VectorXd n = ihla.Normal(i);
        point_normals[i] = n;
        original_normals[i] = point_normals[i];
        inverse_original_normals[i] = -point_normals[i];
    } 
    //
    std::cout<<"normal calculated"<<std::endl;
    // Initialize polyscope
    polyscope::init();   
    polyscope::PointCloud* psCloud = polyscope::registerPointCloud("really great points", P);
    // set some options
    psCloud->setPointRadius(0.01);
    psCloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
    psCloud->addVectorQuantity("Normals", original_normals);
    psCloud->addVectorQuantity("Inverse Normals", inverse_original_normals);
    // show
    polyscope::show();
    return 0;
}
