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
#include <igl/barycenter.h>
#include <igl/boundary_facets.h>
#include <igl/parula.h>
#include <igl/readMESH.h>
#include <igl/slice.h>
#include <igl/marching_tets.h>
#include <igl/winding_number.h>
#include "MeshHelper.h"
#include "iheartmesh.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);


int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    Eigen::MatrixXd BC;
    Eigen::VectorXd W;
    Eigen::MatrixXi T, G;
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("../../../models/small_bunny.obj", meshV, meshF);
    igl::readMESH(argc>1?argv[1]:DATA_PATH / "big-sigcat.mesh", meshV, T, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    igl::barycenter(meshV, T, BC);

      // Compute generalized winding number at all barycenters
    std::cout<<"Computing winding number over all "<<T.rows()<<" tets..."<<std::endl; 
    igl::winding_number(meshV, meshF, BC, W);
    std::cout<<"meshF, rows:"<<meshF.rows()<<", cols:"<<meshF.cols()<<std::endl;
    std::cout<<"BC, rows:"<<BC.rows()<<", cols:"<<BC.cols()<<std::endl;
    std::cout<<"W, rows:"<<W.rows()<<", cols:"<<W.cols()<<std::endl;

    std::cout<<"W, 200:"<<W.row(200)<<std::endl;

      // Extract interior tets
    MatrixXi CT((W.array()>0.5).count(),4);
    {
      size_t k = 0;
      for(size_t t = 0;t<T.rows();t++)
      {
        if(W(t)>0.5)
        {
          CT.row(k) = T.row(t);
          k++;
        }
      }
    }
    // find bounary facets of interior tets
    igl::boundary_facets(CT,G);
    // boundary_facets seems to be reversed...
    G = G.rowwise().reverse().eval();

    // normalize
    W = (W.array() - W.minCoeff())/(W.maxCoeff()-W.minCoeff());



    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshF);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    iheartmesh ihla(triangle_mesh, P);

    Eigen::VectorXd Wind(BC.rows());
    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < BC.rows(); ++i)
    {
        Wind[i] = ihla.w(BC.row(i));
    } 
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    // polyscope::getSurfaceMesh("my mesh")->addVertexDistanceQuantity("GaussianCurvature", gaussian_curvature); 
    polyscope::show();
    return 0;
}
