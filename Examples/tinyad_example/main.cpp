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
#include <igl/readOFF.h>
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
#include <ctime>
#include <chrono>
using namespace autodiff;

int start;
double default_hessian_projection_eps = 1e-9;

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

Eigen::MatrixXd project_positive_definite(
        Eigen::MatrixXd& _H,
        const double& _eigenvalue_eps=default_hessian_projection_eps)
{  
        // std::cout<<"_H is: "<<_H<<std::endl;
        // Compute eigen-decomposition (of symmetric matrix)
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(_H);
        Eigen::MatrixXd D = eig.eigenvalues().asDiagonal();

        // Clamp all eigenvalues to eps
        bool all_positive = true;
        for (Eigen::Index i = 0; i < _H.rows(); ++i)
        {
            if (D(i, i) < _eigenvalue_eps)
            {
                D(i, i) = _eigenvalue_eps;
                all_positive = false;
            }
        }

        // Do nothing if all eigenvalues were already at least eps
        if (all_positive)
            return _H;

        // Re-assemble matrix using clamped eigenvalues
        Eigen::MatrixXd res = eig.eigenvectors() * D * eig.eigenvectors().transpose();
        return res;
}

 
Eigen::SparseMatrix<double > psd(
    const Eigen::MatrixXd & x)
{
    Eigen::SparseMatrix<double > mat = x.sparseView();
    // get local hessian matrix
    Eigen::MatrixXd local_mat = Eigen::MatrixXd::Zero(6, 6);
    std::vector<int> row_map({-1, -1, -1, -1, -1, -1});
    std::vector<int> col_map({-1, -1, -1, -1, -1, -1});
    std::map<int, int> to_local_row_map;
    std::map<int, int> to_local_col_map;
    int cnt = 0;
    int row_index = 0;
    int col_index = 0;
    for (int k=0; k<mat.outerSize(); ++k){
        for (SparseMatrix<double>::InnerIterator it(mat,k); it; ++it){
            int cur_row = it.row();   // row index
            int cur_col = it.col();   // col index (here it is equal to k)
            if (to_local_row_map.find( cur_row ) == to_local_row_map.end() )
            {
                to_local_row_map[cur_row] = row_index;
                row_map[row_index] = cur_row; 
                row_index++;
            }
            if (to_local_col_map.find( cur_col ) == to_local_col_map.end())
            {
                to_local_col_map[cur_col] = col_index;
                col_map[col_index] = cur_col; 
                col_index++;
            } 
            local_mat(to_local_row_map[cur_row], to_local_col_map[cur_col]) = it.value();
            cnt++;
        }
    }
    if (col_index != 6 || row_index != 6)
    {
        std::cout<<"col_index: "<<col_index<<", row_index:"<<row_index<<std::endl;
        Eigen::MatrixXd new_x = x;
        Eigen::MatrixXd res = project_positive_definite(new_x);
        return res.sparseView();
    }
    // assert(col_index == 6 && row_index == 6);
    // std::cout<<"local_mat is:"<<local_mat<<std::endl;
    //
    Eigen::MatrixXd projected_hessian = project_positive_definite(local_mat);
    // std::cout<<"rss is:"<<(projected_hessian-local_mat).norm()<<std::endl;
    // std::cout<<"cnt is:"<<cnt<<std::endl;
    // map to global hessian
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(36);
    for(int i=0; i<6; i++)
    {
        for (int j = 0; j < 6; ++j)
        {
            tripletList.push_back(Eigen::Triplet<double>(row_map[i], col_map[j], projected_hessian(i, j)));
        } 
    }
    Eigen::SparseMatrix<double > proj_mat(mat.rows(), mat.cols());
    proj_mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return proj_mat;  
}

std::vector<Eigen::Matrix<double, 2, 1>> x;
std::vector<Eigen::Matrix<double, 3, 1>> x̄;
Eigen::MatrixXd V; // #V-by-3 3D vertex positions
Eigen::MatrixXd N; // #V-by-3 3D vertex positions
Eigen::MatrixXi F; // #F-by-3 indices into V
Eigen::MatrixXd P; // #V-by-2 3D vertex positions
TriangleMesh triangle_mesh;
double eps = 1e-2;



bool armijo_condition(
        const double _f_curr,
        const double _f_new,
        const double _s,
        const Eigen::VectorXd& _d,
        const Eigen::VectorXd& _g,
        const double _armijo_const)
{
    return _f_new <= _f_curr + _armijo_const * _s * _d.dot(_g);
}

Eigen::VectorXd my_line_search(
        const Eigen::VectorXd& _x0,
        const Eigen::VectorXd& _d,
        const double _f, 
        const Eigen::VectorXd & _g,
        const std::function<double(Eigen::VectorXd&)>& _eval, // Callable of type T(const Eigen::Vector<T, d>&)
        const double _s_max = 1.0, // Initial step size
        const double _shrink = 0.8,
        const int _max_iters = 64,
        const double _armijo_const = 1e-4)
{
    // Check input
    assert( _x0.size() == _g.size());
    if (_s_max <= 0.0)
        std::cout<<"Max step size not positive."<<std::endl;

    // Also try a step size of 1.0 (if valid)
    const bool try_one = _s_max > 1.0;

    Eigen::VectorXd x_new = _x0;
    double s = _s_max;
    for (int i = 0; i < _max_iters; ++i)
    {
        x_new = _x0 + s * _d;
        const double f_new = _eval(x_new);
        std::cout<<"f_new:"<<f_new<<", i:"<<i<<std::endl;
        if (armijo_condition(_f, f_new, s, _d, _g, _armijo_const)){
            std::cout<<"i:"<<i<<std::endl;
            std::cout<<"x_new:"<<x_new<<std::endl;
            return x_new;
        }

        if (try_one && s > 1.0 && s * _shrink < 1.0)
            s = 1.0;
        else
            s *= _shrink;
    }

    std::cout<<"Line search couldn't find improvement. Gradient max norm is " << _g.cwiseAbs().maxCoeff()<<std::endl;
    return _x0;
}


bool step(){
    bool has_updated = true;
    iheartmesh ihla(triangle_mesh, x̄, x, eps, psd);
    std::cout<<"Cur energy is "<<ihla.e<<std::endl;
    Eigen::VectorXd g = ihla.G;
    Eigen::SparseMatrix<double> H = ihla.H;
    // Projected Newton with conjugate gradient solver
    int max_iters = 1000;
    double convergence_eps = 1e-2;
    // Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > cg_solver;
    //
    // Eigen::SparseMatrix<double> id(x.size()*2, x.size()*2);
    // id.setIdentity();
    // Eigen::SparseMatrix<double> param = H + 1e-4 * id;
    // Eigen::VectorXd d = cg_solver.compute(param).solve(-g);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> _solver;
    _solver.analyzePattern(H);
    _solver.factorize(H);
    Eigen::VectorXd d = _solver.solve(-g);


    std::cout<<"H is "<<H<<std::endl; 
    std::cout<<"d is "<<d<<std::endl; 
    Eigen::VectorXd vec_x(x.size()*2);
    for (int i = 0; i < x.size(); ++i)
    {
        vec_x.segment(2*i, 2) = x[i];
    } 
    std::function<double(Eigen::VectorXd&)> energy_func = [&](Eigen::VectorXd& f_x) { 
        std::vector<Eigen::Matrix<double, 2, 1>> v_x(f_x.rows()/2);
        for (int i = 0; i < v_x.size(); ++i)
        {
            v_x[i] = f_x.segment(2*i, 2);
        } 
        double e = 0;
        for(int i : ihla.F){
            e += ihla.S(i, v_x);
        } 
        return e;
    };
    //
    if (-0.5 * d.dot(g) > convergence_eps){
        vec_x = my_line_search(vec_x, d, to_double(ihla.e), g, energy_func);
        std::cout<<"vec_x:"<<vec_x<<std::endl;
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] = vec_x.segment(2*i, 2);
        }
        std::cout<<"x updated"<<std::endl;
    }
    else{
        has_updated = false;
    }
    
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions2D(x);
    return has_updated;
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
        start = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        step();
        auto end = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout <<end-start<< " seconds"<<std::endl;
    } 
    if (ImGui::Button("Five steps")){
        for (int i = 0; i < 5; ++i)
        {
            step();
        }
    } 
    if (ImGui::Button("Run ")){
        for (int i = 0; i < 1000; ++i)
        {
            bool updated = step();
            if (!updated)
            {
                std::cout<<"Finished at i: "<<i<<std::endl;
                break;
            }
        }
    } 
}


int main(int argc, const char * argv[]) {
    // igl::readOBJ("../../../models/armadillo_cut_low.obj", V, F);
    // igl::readOBJ("../../../models/armadillo_cut_low.obj", V, F);
    // igl::readOBJ("../../../models/bunny_cut.obj", V, F);
    // igl::readOBJ("../../../models/snail.obj", V, F);
    // igl::readOBJ("../../../models/camel-head.obj", V, F);
    // igl::readOFF("../../../models/camelhead.off", V, F);
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

    // View resulting parametrization
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", V, F);
    polyscope::state::userCallback = myCallback;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions2D(x);
    polyscope::show();

    return 0;
}
