//
//  main.cpp
//  DEC
//
//  Created by pressure on 10/31/22.
//

#include <iostream>
#include <Eigen/Dense>
#include "Tetrahedron.h"
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/harmonic.h>
#include <igl/boundary_loop.h>
#include <igl/per_vertex_normals.h>
#include <igl/map_vertices_to_circle.h>
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

std::vector<Eigen::Matrix<double, 3, 1>> x;
std::vector<Eigen::Matrix<double, 3, 1>> x̄;
Eigen::MatrixXd V; // #V-by-3 3D vertex positions
// Eigen::MatrixXd T;  
Eigen::MatrixXi F; // #F-by-3 indices into V
// Eigen::MatrixXd VV; // #V-by-3 3D vertex positions
Eigen::MatrixXi T;  
// Eigen::MatrixXi FF; // #F-by-3 indices into V
Tetrahedron tet_mesh;
double eps = 1e-2;
double weight = 1e5;
std::vector<int> bc;
std::vector<Eigen::Matrix<double, 3, 1>> bp;

void step(){
    iheartmesh ihla(tet_mesh, x̄, x, bc, bp, weight, eps);
    std::cout<<"e is "<<ihla.e<<std::endl;
    // std::vector<Eigen::Matrix<double, 2, 1>> y = ihla.y;
    for (int i = 0; i < ihla.y.size(); ++i)
    {
        Eigen::Matrix<double, 3, 1> xx;
        double tmp = ihla.y[i](0,0).expr->val;
        xx(0,0) = tmp;
        tmp = ihla.y[i](1,0).expr->val;
        xx(1,0) = tmp; 
        tmp = ihla.y[i](2,0).expr->val;
        xx(2,0) = tmp; 
        x[i] = xx;
    }
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(x);
}

void myCallback()
{
    if (ImGui::Button("Run/Stop Simulation"))
    {
        step();
    }
}

template<class DT = autodiff::var, class MatrixD = Eigen::Matrix<autodiff::var, Eigen::Dynamic, Eigen::Dynamic>, class VectorD = Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1>>
struct iheartmesh1 {
    DT z;
    Eigen::VectorXd m;
    Eigen::Matrix<DT, 3, 3> A;
    autodiff::ArrayXvar new_A;
    iheartmesh1(
        const double & c,
        const Eigen::Matrix<double, 3, 1> & x,
        const Eigen::Matrix<double, 3, 1> & y,
        const Eigen::Matrix<double, 3, 3> & A)
    {
    
        new_A.resize(3*3);
        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < A.cols(); ++j)
            {
                new_A(i*A.rows()+j) = A(i, j);
                this->A(i, j) = new_A(i*A.rows()+j);
            }
        }
        // z = xᵀ A x
        z = (DT)(x.transpose() * this->A * x);
        // m = ∂z/∂A
        m = gradient(z, this->new_A);
    }
};

void check(){
    double c = rand() % 10;
    Eigen::Matrix<double, 3, 1> x = Eigen::VectorXd::Random(3);
    Eigen::Matrix<double, 3, 1> y = Eigen::VectorXd::Random(3);
    Eigen::Matrix<double, 3, 3> A = Eigen::MatrixXd::Random(3, 3);

    auto z = y.unaryExpr<double(*)(double)>(&std::sin);
    A(0, 1) = A(1, 0);
    A(0, 2) = A(2, 0);
    A(1, 2) = A(2, 1);
    Eigen::Matrix<double, 3, 1> G = 2 * A * x ;
    iheartmesh1 a(c, x, y, A);
    std::cout<<"res: "<<a.m<<std::endl;
    std::cout<<"G: "<<x*x.transpose()<<std::endl;
    std::cout<<"z: "<<z<<std::endl;
}


int main(int argc, const char * argv[]) {
    srand((int)time(NULL));
    // igl::readOBJ("../../../../models/armadillo_cut_low.obj", V, F);
    // igl::readOBJ("../../../models/cube.obj", V, F);
    check();
    return 0;

    // igl::copyleft::tetgen::tetrahedralize(V, F,"pq1.414Y", VV, TT, FF);

    igl::readMESH("../../../models/cube_26v.1.mesh", V, T, F);
    // std::cout<<"V rows:"<<VV.rows()<<std::endl;
    // std::cout<<"TT rows:"<<TT.rows()<<", cols:"<<TT.cols()<<std::endl;
    // std::cout<<"TT:"<<TT<<std::endl;
    // std::cout<<"FF rows:"<<FF.rows()<<", cols:"<<FF.cols()<<std::endl;
    // Eigen::MatrixXd P = tutte_embedding(V, F); // #V-by-2 2D vertex positions

    tet_mesh.initialize(T);
    bc.push_back(0);
    bc.push_back(10);

    Eigen::Matrix<double, 3, 1> x0;
    x0 << -1, -1, 0.7;
    Eigen::Matrix<double, 3, 1> x10;
    x10 << -0.75, -0.25, 0.65; 
    Eigen::Matrix<double, 3, 1> x24;
    x24 << 1, 0, 0;
    bp.push_back(x0);
    bp.push_back(x10);
    std::cout<<"V is:"<<V<<std::endl;
    for (int i = 0; i < V.rows(); ++i)
    {
        x.push_back(V.row(i));
        x̄.push_back(V.row(i));
    }
 
    // x[0] = x0;
    // x[10] = x10;



    // std::cout<<"x.size is:"<<x.size()<<std::endl;
    // std::cout<<"x̄.size is:"<<x̄.size()<<std::endl;

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



    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", V, F);
    polyscope::state::userCallback = myCallback;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(x);

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
