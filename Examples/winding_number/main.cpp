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
#include <igl/slice_into.h>
#include <igl/marching_tets.h>
#include <igl/winding_number.h>
#include <igl/opengl/glfw/Viewer.h>
#include "MeshHelper.h"
#include "iheartmesh.h"
#include "dec_util.h"
#include <filesystem>
namespace fs = std::filesystem;
using namespace Eigen;
using namespace std;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

Eigen::MatrixXd V,BC; 
Eigen::VectorXd W;
Eigen::MatrixXi T,F,G;
RowVector3d f_c(0.486,0.549,0.815);   // face color
double slice_z = 0.5;
enum OverLayType
{
  OVERLAY_NONE = 0,
  OVERLAY_INPUT = 1,
  OVERLAY_OUTPUT = 2,
  NUM_OVERLAY = 3,
} overlay = OVERLAY_NONE;

void update_visualization(igl::opengl::glfw::Viewer & viewer)
{
    Eigen::Vector4d plane(
    0,0,1,-((1-slice_z)*V.col(2).minCoeff()+slice_z*V.col(2).maxCoeff()));
    MatrixXd V_vis;
    MatrixXi F_vis;
    VectorXi J;
    {
        SparseMatrix<double> bary;
        // Value of plane's implicit function at all vertices
        const VectorXd IV = 
          (V.col(0)*plane(0) + 
            V.col(1)*plane(1) + 
            V.col(2)*plane(2)).array()
          + plane(3);
        igl::marching_tets(V,T,IV,V_vis,F_vis,J,bary);
    }
    VectorXd W_vis;
    igl::slice(W,J,W_vis);
    MatrixXd C_vis = f_c.replicate(W_vis.rows(),1);
    const auto & append_mesh = [&C_vis,&F_vis,&V_vis](
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const RowVector3d & color)
    {
        F_vis.conservativeResize(F_vis.rows()+F.rows(),3);
        F_vis.bottomRows(F.rows()) = F.array()+V_vis.rows();
        V_vis.conservativeResize(V_vis.rows()+V.rows(),3);
        V_vis.bottomRows(V.rows()) = V;
        C_vis.conservativeResize(C_vis.rows()+F.rows(),3);
        C_vis.bottomRows(F.rows()).rowwise() = color;
    };
    switch(overlay)
    {
        case OVERLAY_INPUT:
          append_mesh(V,F, f_c);
          break;
        case OVERLAY_OUTPUT:
          append_mesh(V,G,RowVector3d(0.8,0.8,0.8));
          break;
        default:
          break;
    }
    viewer.data().clear();
    viewer.data().set_mesh(V_vis,F_vis);
    viewer.data().set_colors(C_vis);
    viewer.data().set_face_based(true);
    viewer.core().background_color.setConstant(1);
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int mod)
{
    switch(key)
    {
        default:
          return false;
        case ' ':
          overlay = (OverLayType)((1+(int)overlay)%NUM_OVERLAY);
          break;
        case '.':
          slice_z = std::min(slice_z+0.01,0.99);
          break;
        case ',':
          slice_z = std::max(slice_z-0.01,0.01);
          break;
    }
    update_visualization(viewer);
    return true;
}

void set_winding_number(){
    // igl::winding_number(V,F,BC,W);   // libigl version
    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(F);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < V.rows(); ++i)
    {
        P.push_back(V.row(i).transpose());
    }
    iheartmesh ihla(triangle_mesh, P);
    int start = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        ;
    W.resize(BC.rows());
    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < BC.rows(); ++i)
    {
        W[i] = ihla.w(BC.row(i));
    } 
    auto end = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout <<end-start<< " seconds int total"<<std::endl;
    cout<<"W, rows:"<<W.rows()<<", cols:"<<W.cols()<<endl;
}


void set_slicing_colors(){
    // Compute barycenters of all tets
    igl::barycenter(V,T,BC);
    // Compute generalized winding number at all barycenters
    cout<<"Computing winding number over all "<<T.rows()<<" tets..."<<endl;
    set_winding_number();
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
}

int main(int argc, char *argv[])
{

  cout<<"Usage:"<<endl;
  cout<<"[space]  toggle showing input mesh, output mesh or slice "<<endl;
  cout<<"         through tet-mesh of convex hull."<<endl;
  cout<<"'.'/','  push back/pull forward slicing plane."<<endl;
  cout<<endl;
  // Load mesh: (V,T) tet-mesh of convex hull, F contains facets of input
  // surface mesh _after_ self-intersection resolution
  // igl::readMESH(TUTORIAL_SHARED_PATH "/big-sigcat.mesh",V,T,F);
  igl::readMESH(argc>1?argv[1]:DATA_PATH / "holy-cow.mesh", V, T, F);  //6851 sec
  set_slicing_colors();
  // Plot the generated mesh
  igl::opengl::glfw::Viewer viewer;
  update_visualization(viewer);
  viewer.callback_key_down = &key_down; 
  viewer.launch();
  viewer.core().background_color.setConstant(1);
}
