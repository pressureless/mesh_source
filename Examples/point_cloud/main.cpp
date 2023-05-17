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
#include "heartlib.h"
#include "orientation.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "PointCloud.h"
#include <LBFGS.h>
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

using namespace iheartmesh;
using namespace LBFGSpp;

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    igl::readOBJ(argc>1?argv[1]:DATA_PATH / "small_bunny.obj", meshV, meshF);
    // igl::readOBJ(argc>1?argv[1]:DATA_PATH / "sphere3.obj", meshV, meshF);
    // igl::readOBJ("../../../models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    std::vector<Eigen::VectorXd> PN;
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        PN.push_back(meshV.row(i).transpose());
        P.push_back(meshV.row(i).transpose());
    }

    // std::vector<std::vector<size_t>> neighbors = GetPointNeighbors(PN, 6);
    std::vector<std::vector<size_t>> neighbors = GetPointNeighbors(PN, 4);
    // std::vector<std::vector<size_t>> neighbors = GetPointNeighbors(PN, 0.2);
    PointCloud pc(PN, neighbors);
    heartlib ihla(pc, P);
    std::vector<Eigen::Matrix<double, 3, 3>> cov;
    // std::cout<<"v:"<<meshV.rows()<<std::endl;
    std::vector<Eigen::VectorXd> point_normals(meshV.rows());

    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Eigen::VectorXd n = ihla.Normal(i);
        // Naive way to check whether we need to flip the normal (for convex .obj)
        // if (P[i].dot(n) < 0)
        // {
        //     n = -n;
        // }
        point_normals[i] = n/n.norm();
    } 
    //
    std::cout<<"normal end:"<<std::endl;
    // Set up parameters
    LBFGSParam<double> param;
    param.epsilon = 1e-10;
    param.max_iterations = 100;
    // Create solver and function object
    LBFGSpp::LBFGSSolver<double> solver(param);
    // Initial guess
    Eigen::VectorXd s = Eigen::VectorXd::Random(point_normals.size());
    std::cout << "s = " << s.transpose() << std::endl;
    // x will be overwritten to be the best point found
    auto func = [&]( const Eigen::VectorXd& S, Eigen::VectorXd& gradient_out ) -> double {
            orientation ori(pc, S, point_normals);
            gradient_out = ori.g;
            std::cout << "energy: "<<ori.total<< std::endl;
            return to_double(ori.total);
    };
    double fx;
    int niter = solver.minimize(func, s, fx);
    std::cout << niter << " iterations" << std::endl;
    std::cout << "s = \n" << s.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;

    for (int i = 0; i < meshV.rows(); ++i)
    {
        point_normals[i] = point_normals[i] * (s(i)>0?1:-1);
    } 
    // Initialize polyscope
    polyscope::init();   
    polyscope::PointCloud* psCloud = polyscope::registerPointCloud("really great points", P);
    // set some options
    psCloud->setPointRadius(0.01);
    psCloud->setPointRenderMode(polyscope::PointRenderMode::Sphere);
    psCloud->addVectorQuantity("Normals", point_normals);
    // show
    polyscope::show();
    return 0;
}
