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
#include <igl/readSTL.h>
#include <igl/harmonic.h>
#include <igl/boundary_loop.h>
#include <igl/per_vertex_normals.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/boundary_loop.h>
#include <igl/per_vertex_normals.h>
#include <igl/map_vertices_to_circle.h>

// #include <igl/opengl/glfw/Viewer.h>
#include "MeshHelper.h"
#include "iheartmesh.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

inline Eigen::MatrixXd tutte_embedding(
    const Eigen::MatrixXd& _V,
    const Eigen::MatrixXi& _F)
{
  Eigen::VectorXi b; // #constr boundary constraint indices
  Eigen::MatrixXd bc; // #constr-by-2 2D boundary constraint positions
  Eigen::MatrixXd P; // #V-by-2 2D vertex positions
  igl::boundary_loop(_F, b); // Identify boundary vertices
  igl::map_vertices_to_circle(_V, b, bc); // Set boundary vertex positions
  igl::harmonic(_F, b, bc, 1, P); // Compute interior vertex positions

  return P;
}

// double& operator +(double& lhs, const autodiff::var& rhs ) { 
//     std::cout<<"called"<<std::endl;
//     lhs = rhs.expr->val; 
//     return lhs; 
// }


template <typename Derived>
auto col_mat(
        const Eigen::MatrixBase<Derived>& _v0,
        const Eigen::MatrixBase<Derived>& _v1)
{
    using T = typename Derived::Scalar;
    Eigen::Matrix<T, Derived::RowsAtCompileTime, 2 * Derived::ColsAtCompileTime> M;

    M << _v0, _v1;

    return M;
}

/**
 * Assemble matrix from column vectors.
 */
template <typename Derived>
auto col_mat(
        const Eigen::MatrixBase<Derived>& _v0,
        const Eigen::MatrixBase<Derived>& _v1,
        const Eigen::MatrixBase<Derived>& _v2)
{
    using T = typename Derived::Scalar;
    Eigen::Matrix<T, Derived::RowsAtCompileTime, 3 * Derived::ColsAtCompileTime> M;

    M << _v0, _v1, _v2;

    return M;
}

// void test(){
//     Eigen::MatrixXi Tet(2,3);
//     Tet <<
//     0,1,2,
//     2,1,3; 
//     TriangleMesh triangle_mesh_0;
//     triangle_mesh_0.initialize(Tet);

//     std::vector<Eigen::Matrix<double, 2, 1>> P;
//     Eigen::Matrix<double, 2, 1> P0;
//     P0 << 0, 1;
//     Eigen::Matrix<double, 2, 1> P1;
//     P1 << 1, 0;
//     Eigen::Matrix<double, 2, 1> P2;
//     P2 << 1, 2;
//     Eigen::Matrix<double, 2, 1> P3;
//     P3 << 2, 1; 
//     P.push_back(P0);
//     P.push_back(P1);
//     P.push_back(P2);
//     P.push_back(P3);
//     //
//     std::vector<Eigen::Matrix<double, 2, 1>> x;
//     Eigen::Matrix<double, 2, 1> x0;
//     x0 << 0.2, 1;
//     Eigen::Matrix<double, 2, 1> x1;
//     x1 << 1.3, 0;
//     Eigen::Matrix<double, 2, 1> x2;
//     x2 << 1, 2.1;
//     Eigen::Matrix<double, 2, 1> x3;
//     x3 << 2.4, 1.1; 
//     x.push_back(x0);
//     x.push_back(x1);
//     x.push_back(x2);
//     x.push_back(x3);

//     iheartmesh ihla(triangle_mesh_0, P);

//     double energy = ihla.Energy(x);

//     // vertex one ring
//     std::cout<<"energy: "<<energy<<std::endl;
// }

std::vector<Eigen::Matrix<double, 2, 1>> x;
std::vector<Eigen::Matrix<double, 3, 1>> x̄;
Eigen::MatrixXd V; // #V-by-3 3D vertex positions
Eigen::MatrixXd N; // #V-by-3 3D vertex positions
Eigen::MatrixXi F; // #F-by-3 indices into V
Eigen::MatrixXd P; // #V-by-2 3D vertex positions
TriangleMesh triangle_mesh;
double eps = 1e-2;

void step(){
    iheartmesh ihla(triangle_mesh, x̄, x, eps);
    std::cout<<"e is "<<ihla.e<<std::endl;
    // std::vector<Eigen::Matrix<double, 2, 1>> y = ihla.y;
    // x = to_double(ihla.y);
    for (int i = 0; i < x.size(); ++i)
    {
        Eigen::Matrix<double, 2, 1> xx;
        double tmp = ihla.y[i](0,0).expr->val;
        xx(0,0) = tmp;
        tmp = ihla.y[i](1,0).expr->val;
        xx(1,0) = tmp; 
        x[i] = xx;
    }
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions2D(x);
}

void myCallback()
{
    if (ImGui::Button("Run/Stop Simulation"))
    {
        std::cout<<"Run stop"<<std::endl;
    }
    // ImGui::SameLine();
    if (ImGui::Button("One step")){
        std::cout<<"one step"<<std::endl;
        step();
    } 
    if (ImGui::Button("Five steps")){
        for (int i = 0; i < 5; ++i)
        {
            step();
        }
    } 
}


int main(int argc, const char * argv[]) {
    // igl::readOBJ("../../../../models/armadillo_cut_low.obj", V, F);
    // igl::readOBJ("../../../models/armadillo_cut_low.obj", V, F);
    // igl::readOBJ("../../../models/bunny_cut.obj", V, F);
    // igl::readOBJ("../../../models/snail.obj", V, F);
    // igl::readOBJ("../../../models/camel-head.obj", V, F);
    igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/camel-head_54.obj", V, F);
    // std::ifstream input_file("../../../models/1004826.stl");
    // igl::readSTL(input_file, V, F, N);
    P = tutte_embedding(V, F); // #V-by-2 2D vertex positions

    triangle_mesh.initialize(F);
    // std::cout<<"P is:"<<P<<std::endl;
    for (int i = 0; i < P.rows(); ++i)
    {
        x.push_back(P.row(i));
        x̄.push_back(V.row(i));
    }
    std::cout<<"x.size is:"<<x.size()<<std::endl;
    std::cout<<"x̄.size is:"<<x̄.size()<<std::endl;

    // std::vector<Eigen::Matrix<double, 2, 1>> P;
    // Eigen::Matrix<double, 2, 1> P0;
    // P0 << 0.2, 1;
    // Eigen::Matrix<double, 2, 1> P1;
    // P1 << 1, 0.4;
    // Eigen::Matrix<double, 2, 1> P2;
    // P2 << 1.3, 2;
    // Eigen::Matrix<double, 2, 1> P3;
    // P3 << 2, 1.4; 
    // P.push_back(P0);
    // P.push_back(P1);
    // P.push_back(P2);
    // P.push_back(P3);

    // std::vector<Eigen::Matrix<double, 2, 1>> x;
    // std::vector<Eigen::Matrix<double, 2, 1>> x̄;
    // for (int i = 0; i < P.size(); ++i)
    // {
    //     x.push_back(P[i]);
    //     x̄.push_back(P[i]);
    // }

    // Eigen::MatrixXi Tet(2,3);
    // Tet <<
    // 0,1,2,
    // 2,1,3; 



    // Pre-compute triangle rest shapes in local coordinate systems
    // std::vector<Eigen::Matrix2d> rest_shapes(F.rows());
    // std::vector<Eigen::Matrix2d> updated_shapes(F.rows());
    // for (int f_idx = 0; f_idx < F.rows(); ++f_idx)
    // {
    //     // Get 3D vertex positions
    //     Eigen::Vector3d ar_3d = V.row(F(f_idx, 0));
    //     Eigen::Vector3d br_3d = V.row(F(f_idx, 1));
    //     Eigen::Vector3d cr_3d = V.row(F(f_idx, 2));

    //     // Set up local 2D coordinate system
    //     Eigen::Vector3d n = (br_3d - ar_3d).cross(cr_3d - ar_3d);
    //     Eigen::Vector3d b1 = (br_3d - ar_3d).normalized();
    //     Eigen::Vector3d b2 = n.cross(b1).normalized();

    //     // Express a, b, c in local 2D coordiante system
    //     Eigen::Vector2d ar_2d(0.0, 0.0);
    //     Eigen::Vector2d br_2d((br_3d - ar_3d).dot(b1), 0.0);
    //     Eigen::Vector2d cr_2d((cr_3d - ar_3d).dot(b1), (cr_3d - ar_3d).dot(b2));

    //     // Save 2-by-2 matrix with edge vectors as colums
    //     rest_shapes[f_idx] = col_mat(br_2d - ar_2d, cr_2d - ar_2d);
    //     updated_shapes[f_idx] = rest_shapes[f_idx];
    // };
    // for (int i = 0; i < 3; ++i)
    // {
    //     step();
    // }
    
    // var ff = 2;
    // double cc = to_double(ff);


    // double aaa = to_dt<double>(ff);
    // var gbb = to_dt<var>(aaa);
    // std::cout<<"aaa is "<<aaa<<std::endl;
    // std::cout<<"gbb is "<<gbb<<std::endl;
    // test();
    // View resulting parametrization
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", V, F);
    polyscope::state::userCallback = myCallback;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions2D(x);

    // update();
    // iheartmesh ihla(triangle_mesh, P); 
    // // std::cout<<"before"<<std::endl;

    // std::cout<<"nn:\n"<<std::endl;
    // i, j, k = ihla.Vertices(0);
    // // print_set();
    // std::cout<<"i:"<<i<<", j:"<<j<<", k:"<<k<<std::endl;
    // std::cout<<"nn:\n"<<P[0]<<std::endl;
    // polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("VertexNormal", N); 
    polyscope::show();

    return 0;
}
