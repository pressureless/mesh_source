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
#include <igl/per_vertex_normals.h>

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
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);
using namespace autodiff;

int start;
double default_hessian_projection_eps = 1e-9;
std::vector<Eigen::Matrix<double, 2, 1>> x;
std::vector<Eigen::Matrix<double, 3, 1>> x̄;
Eigen::MatrixXd V; // #V-by-3 3D vertex positions
Eigen::MatrixXd N; // #V-by-3 3D vertex positions
Eigen::MatrixXi F; // #F-by-3 indices into V
Eigen::MatrixXd P; // #V-by-2 3D vertex positions
TriangleMesh triangle_mesh;
double eps = 1e-2;

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

Eigen::MatrixXd project_positive_definite(
        Eigen::MatrixXd& _H,
        const double& _eigenvalue_eps=default_hessian_projection_eps)
{  
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
    std::vector<int> dim_map({-1, -1, -1, -1, -1, -1});
    std::map<int, int> to_local_map;
    int cnt = 0;
    int dim_index = 0;
    for (int k=0; k<mat.outerSize(); ++k){
        for (SparseMatrix<double>::InnerIterator it(mat,k); it; ++it){
            int cur_row = it.row();   // row index
            int cur_col = it.col();   // col index (here it is equal to k)
            if (to_local_map.find( cur_row ) == to_local_map.end() )
            {
                to_local_map[cur_row] = dim_index;
                dim_map[dim_index] = cur_row; 
                dim_index++;
            }
            if (to_local_map.find( cur_col ) == to_local_map.end())
            {
                to_local_map[cur_col] = dim_index;
                dim_map[dim_index] = cur_col; 
                dim_index++;
            } 
            local_mat(to_local_map[cur_row], to_local_map[cur_col]) = it.value();
            cnt++;
        }
    }
    if (dim_index != 6)
    {
        std::cout<<"dim_index: "<<dim_index<<std::endl;
        Eigen::MatrixXd new_x = x;
        Eigen::MatrixXd res = project_positive_definite(new_x);
        return res.sparseView();
    }
    //
    Eigen::MatrixXd projected_hessian = project_positive_definite(local_mat);
    // map to global hessian
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(36);
    for(int i=0; i<6; i++)
    {
        for (int j = 0; j < 6; ++j)
        {
            tripletList.push_back(Eigen::Triplet<double>(dim_map[i], dim_map[j], projected_hessian(i, j)));
        } 
    }
    Eigen::SparseMatrix<double > proj_mat(mat.rows(), mat.cols());
    proj_mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return proj_mat;  
}

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
        if (armijo_condition(_f, f_new, s, _d, _g, _armijo_const)){
            // std::cout<<"i:"<<i<<std::endl;
            // std::cout<<"x_new:"<<x_new<<std::endl;
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
    iheartmesh ihla(triangle_mesh, x̄, x, eps, psd, INFINITY);
    std::cout<<"Cur energy is "<<ihla.e<<std::endl;
    Eigen::VectorXd g = ihla.G;
    Eigen::SparseMatrix<double> H = ihla.H;
    // Projected Newton w
    double convergence_eps = 1e-2;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> _solver;
    _solver.analyzePattern(H);
    _solver.factorize(H);
    Eigen::VectorXd d = _solver.solve(-g);
    //
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
        // std::cout<<"vec_x:"<<vec_x<<std::endl;
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] = vec_x.segment(2*i, 2);
        }
        // std::cout<<"x updated"<<std::endl;
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
        std::cout <<end-start<< " seconds for current step"<<std::endl;
    } 
    if (ImGui::Button("Five steps")){
        for (int i = 0; i < 5; ++i)
        {
            step();
        }
    } 
    if (ImGui::Button("Run ")){
        start = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        for (int i = 0; i < 1000; ++i)
        {
            bool updated = step();
            if (!updated)
            {
                std::cout<<"Finished at i: "<<i<<std::endl;
                break;
            }
            auto end = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            std::cout<<"cur:"<<i<<", time:" <<end-start<< " seconds"<<std::endl;
        }
    } 
}


int main(int argc, const char * argv[]) {
    // igl::readOBJ("../../../models/armadillo_cut_low.obj", V, F);
    // igl::readOBJ("../../../models/armadillo_cut_low.obj", V, F);
    // igl::readOBJ("../../../models/bunny_cut.obj", V, F);
    // igl::readOBJ("../../../models/snail.obj", V, F);
    // igl::readOBJ(DATA_PATH / "camel-head.obj", V, F); 
    igl::readOBJ(DATA_PATH / "camelhead-decimate-qslim.obj", V, F); 
    // igl::readOBJ(DATA_PATH / "animal-straightened-decimated.obj", V, F); 
    // igl::readOFF("../../../models/camelhead.off", V, F);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/camel-head_54.obj", V, F);
    // std::ifstream input_file("../../../models/1004826.stl");
    // igl::readSTL(input_file, V, F, N);
    Eigen::MatrixXd N;
    igl::per_vertex_normals(V,F,N); 
    P = tutte_embedding(V, F); // #V-by-2 2D vertex positions

    triangle_mesh.initialize(F);
    // std::cout<<"P is:"<<P<<std::endl;
    for (int i = 0; i < P.rows(); ++i)
    {
        x.push_back(P.row(i));
        x̄.push_back(V.row(i));
    }
    // View resulting parametrization
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", V, F);
    polyscope::state::userCallback = myCallback;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions2D(x);   // original parameterization 
    polyscope::getSurfaceMesh("my mesh")->addVertexColorQuantity("vColor", N.array()*0.5+0.5);
    // polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(x̄);  // original 3D shape
    polyscope::show();

    return 0;
}
