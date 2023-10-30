#include <iostream>
#include <Eigen/Dense>
#include "Tetrahedron.h"
#include "CellMesh.h"
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include <igl/readMESH.h>
#include <igl/writeMESH.h>
#include <igl/harmonic.h>
#include <igl/boundary_loop.h>
#include <igl/per_vertex_normals.h>
#include <igl/map_vertices_to_circle.h>
#include "MeshHelper.h"
#include "heartlib.h"
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

using namespace iheartmesh;

int start;
double default_hessian_projection_eps = 1e-9;
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
    // return x.sparseView();
    Eigen::SparseMatrix<double > mat = x.sparseView();
    int max_dims = 108;
    // get local hessian matrix
    Eigen::MatrixXd local_mat = Eigen::MatrixXd::Zero(max_dims, max_dims);
    std::vector<int> dim_map;
    dim_map.reserve(max_dims);
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
    local_mat.conservativeResize(dim_index, dim_index);
    Eigen::MatrixXd projected_hessian = project_positive_definite(local_mat);
    // map to global hessian
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(max_dims*max_dims);
    for(int i=0; i<local_mat.rows(); i++)
    {
        for (int j = 0; j < local_mat.cols(); ++j)
        {
            if (isnan(projected_hessian(i, j)))
            {
                std::cout<<"hessian isnan, i:"<<dim_map[i]<<", j:"<<dim_map[j]<<std::endl;
            }
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
        // std::cout<<"f_new:"<<f_new<<", i:"<<i<<std::endl;
        if (armijo_condition(_f, f_new, s, _d, _g, _armijo_const)){
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


std::vector<Eigen::Matrix<double, 3, 1>> x;
std::vector<Eigen::Matrix<double, 3, 1>> x̄;
Eigen::MatrixXd V; // #V-by-3 3D vertex positions
// Eigen::MatrixXd T;  
Eigen::MatrixXi F; // #F-by-3 indices into V
// Eigen::MatrixXd VV; // #V-by-3 3D vertex positions
Eigen::MatrixXi T;  
// Eigen::MatrixXi FF; // #F-by-3 indices into V
Tetrahedron tet_mesh;
CellMesh *cell_mesh;
double eps = 1e-2;
double weight = 1e5;
std::vector<int> bc;
std::vector<Eigen::Matrix<double, 3, 1>> bp;

bool step(){
    bool has_updated = true;
    heartlib ihla(*cell_mesh, x̄, x, bc, bp, weight, eps, psd);

    auto end = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout <<end-start<< " seconds init"<<std::endl;
    std::cout<<"Cur energy is "<<ihla.e<<std::endl;
    Eigen::VectorXd g = ihla.G;
    Eigen::SparseMatrix<double> H = ihla.H;
    // Projected Newton with conjugate gradient solver
    int max_iters = 1000;

    double convergence_eps = 1e-4;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg_solver;
    //
    Eigen::SparseMatrix<double> _id(x.size()*3, x.size()*3);
    _id.setIdentity();
    Eigen::SparseMatrix<double> param = H + 1e-4 * _id;
    Eigen::VectorXd d = cg_solver.compute(param).solve(-g);
    Eigen::VectorXd vec_x(x.size()*3);
    for (int i = 0; i < x.size(); ++i)
    {
        vec_x.segment(3*i, 3) = x[i];
    } 
    std::function<double(Eigen::VectorXd&)> energy_func = [&](Eigen::VectorXd& f_x) { 
        std::vector<Eigen::Matrix<double, 3, 1>> v_x(f_x.rows()/2);
        for (int i = 0; i < v_x.size(); ++i)
        {
            v_x[i] = f_x.segment(3*i, 3);
        } 
        double e = 0;
        for(int i : ihla.C){
            e += ihla.S(i, v_x);
            e += ihla.CAMIPS(i, v_x);
        } 
        return e;
    };
    //
    std::cout<<"threshold is  "<<-0.5 * d.dot(g)<<std::endl; 
    if (-0.5 * d.dot(g) > convergence_eps){
        vec_x = my_line_search(vec_x, d, to_double(ihla.e), g, energy_func);
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] = vec_x.segment(3*i, 3);
        }
        std::cout<<"x updated:"<<std::endl;
    }
    else{
        has_updated = false;
    }
     
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(x);
    return has_updated;
}

void myCallback()
{
    if (ImGui::Button("One step"))
    {
        start = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        step();
        auto end = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout <<end-start<< " seconds for current step"<<std::endl;
    }
    if (ImGui::Button("Run/Stop Simulation"))
    {
        start = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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


void load_cube_tet(){
    igl::readMESH(DATA_PATH /  "cube_v729.mesh", V, T, F);
    tet_mesh.initialize(T);
    cell_mesh = new CellMesh(tet_mesh.bm1, tet_mesh.bm2, tet_mesh.bm3);
    for (int i = 0; i < V.rows(); ++i)
    {
        x.push_back(V.row(i));
        x̄.push_back(V.row(i));
    }
    //
    int v_cnt = 36;
    Eigen::MatrixXd line_pos(v_cnt, 3); 
    line_pos<< 0.4,   0.65,   -0.1,   // line 1
                0.3, 0.65,   -0.1,
                0.2,  0.65,   -0.1,
                0.1,  0.65,   -0.1,
                0,  0.65,   -0.1,
                -0.1,  0.65,   -0.1,
                -0.2,  0.65,   -0.1,
                -0.3,  0.65,   -0.1,
                -0.4,  0.65,   -0.1,

                0.2,   -0.9,   -0.3,       //  line 2
                0.15, -0.925,   -0.35,
                0.1,  -0.95,   -0.4,
                0.05,  -0.975,   -0.45,
                0,  -1,   -0.5,
                -0.05,  -1.025,   -0.55,
                -0.1,  -1.05,   -0.6,
                -0.15,  -1.075,   -0.65,
                -0.2,  -1.1,   -0.7, 

                0.5,   0.8,   0.52,    //  line 3
                0.375, 0.75,   0.52,
                0.25,  0.7,   0.52,
                0.125,  0.65,   0.52,
                0,  0.6,   0.52,
                -0.125,  0.55,   0.52,
                -0.25,  0.5,   0.52,
                -0.375,  0.45,   0.52,
                -0.5,  0.4,   0.52,

                0.5,   -0.51,   0.4,   //   line 4
                0.375, -0.51,   0.45,
                0.25,  -0.51,   0.5,
                0.125,  -0.51,   0.55,
                0,  -0.51,   0.6,
                -0.125,  -0.51,   0.65,
                -0.25,  -0.51,   0.7,
                -0.375,  -0.51,   0.75,
                -0.5,  -0.51,   0.8;
    Eigen::VectorXi bcc(v_cnt); bcc<< 80, 79, 78, 77, 76, 75, 74, 73, 72,
    8, 7, 6, 5, 4, 3, 2, 1, 0,
    728, 727, 726, 725, 724, 723, 722, 721, 720,
    656, 655,  654,  653,  652,  651,  650, 649, 648;
    bc.resize(v_cnt);
    for (int i = 0; i < v_cnt; ++i)
    {
        bc[i] = bcc(i);
    }
    bp.resize(line_pos.rows());
    for (int i = 0; i < v_cnt; ++i)
    {
        bp[i] = line_pos.row(i);
    }
}


int main(int argc, const char * argv[]) {
    srand((int)time(NULL));

    load_cube_tet();

    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", V, F);
    polyscope::state::userCallback = myCallback;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(x);
    polyscope::show();
    return 0;
}
