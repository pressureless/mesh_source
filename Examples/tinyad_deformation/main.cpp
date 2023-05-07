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
#include <ctime>
#include <chrono>
using namespace autodiff;

double default_hessian_projection_eps = 1e-9;
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
    return mat;
    // get local hessian matrix
    Eigen::MatrixXd local_mat = Eigen::MatrixXd::Zero(12, 12);
    std::vector<int> row_map({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
    std::vector<int> col_map({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
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
    if (col_index != 12 || row_index != 12)
    {
        std::cout<<"col_index: "<<col_index<<", row_index:"<<row_index<<std::endl;
        Eigen::MatrixXd new_x = x;
        Eigen::MatrixXd res = project_positive_definite(new_x);
        return res.sparseView();
        return mat;  
    }
    // assert(col_index == 6 && row_index == 6);
    // std::cout<<"local_mat is:"<<local_mat<<std::endl;
    //
    Eigen::MatrixXd projected_hessian = project_positive_definite(local_mat);
    // std::cout<<"rss is:"<<(projected_hessian-local_mat).norm()<<std::endl;
    // std::cout<<"cnt is:"<<cnt<<std::endl;
    // map to global hessian
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(144);
    for(int i=0; i<12; i++)
    {
        for (int j = 0; j < 12; ++j)
        {
            tripletList.push_back(Eigen::Triplet<double>(row_map[i], col_map[j], projected_hessian(i, j)));
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

bool step(){
    bool has_updated = true;
    iheartmesh ihla(tet_mesh, x̄, x, bc, bp, weight, eps, psd);
    std::cout<<"Cur energy is "<<ihla.e<<std::endl;
    Eigen::VectorXd g = ihla.G;
    Eigen::SparseMatrix<double> H = ihla.H;
    // Projected Newton with conjugate gradient solver
    int max_iters = 1000;

    double convergence_eps = 1e-4;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg_solver;
    //
    Eigen::SparseMatrix<double> id(x.size()*3, x.size()*3);
    id.setIdentity();
    Eigen::SparseMatrix<double> param = H + 1e-4 * id;
    Eigen::VectorXd d = cg_solver.compute(param).solve(-g);


    std::cout<<"H is "<<H<<std::endl; 
    // std::cout<<"d is "<<d<<std::endl; 
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
        } 
        return e;
    };
    //
    if (-0.5 * d.dot(g) > convergence_eps){
    // std::cout<<"threshold is  "<<-0.5 * d.dot(g)<<std::endl; 
        vec_x = my_line_search(vec_x, d, to_double(ihla.e), g, energy_func);
        // std::cout<<"vec_x:"<<vec_x<<std::endl;
        for (int i = 0; i < x.size(); ++i)
        {
            x[i] = vec_x.segment(3*i, 3);
        }
        // std::cout<<"x updated:"<<vec_x<<std::endl;
    }
    else{
        has_updated = false;
    }
     
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(x);
    return has_updated;
}

void myCallback()
{
    if (ImGui::Button("Run/Stop Simulation"))
    {
        step();
    }
}


int main(int argc, const char * argv[]) {
    srand((int)time(NULL));
    // igl::readOBJ("../../../../models/armadillo_cut_low.obj", V, F);
    // igl::readOBJ("../../../models/cube.obj", V, F);
    // check();
    // return 0;
    // igl::copyleft::tetgen::tetrahedralize(V, F,"pq1.414Y", VV, TT, FF);
    igl::readMESH("../../../models/cube_26v.1.mesh", V, T, F);
    // std::cout<<"V rows:"<<VV.rows()<<std::endl;
    // std::cout<<"TT rows:"<<TT.rows()<<", cols:"<<TT.cols()<<std::endl;
    // std::cout<<"TT:"<<TT<<std::endl;
    // std::cout<<"FF rows:"<<FF.rows()<<", cols:"<<FF.cols()<<std::endl;
    // Eigen::MatrixXd P = tutte_embedding(V, F); // #V-by-2 2D vertex positions

    // igl::readOBJ("../../../models/cube_48k.obj", V, T);
    // Eigen::VectorXi bbc(84); 
    // bbc << 440, 881, 1322, 1763, 2204, 2645, 3086, 3527, 3968, 4409, 4850, 5291, 5732, 6173, 6614, 7055, 7496, 7937, 8378, 8819, 9260, 0, 441, 882, 1323, 1764, 2205, 2646, 3087, 3528, 3969, 4410, 4851, 5292, 5733, 6174, 6615, 7056, 7497, 7938, 8379, 8820, 20, 461, 902, 1343, 1784, 2225, 2666, 3107, 3548, 3989, 4430, 4871, 5312, 5753, 6194, 6635, 7076, 7517, 7958, 8399, 8840, 420, 861, 1302, 1743, 2184, 2625, 3066, 3507, 3948, 4389, 4830, 5271, 5712, 6153, 6594, 7035, 7476, 7917, 8358, 8799, 9240;
    // bc.resize(84);
    // for (int i = 0; i < bbc.rows(); ++i)
    // {
    //     bc[i] = bbc(i);
    // }
    // Eigen::MatrixXd bbp(84, 3); 
    // bbp << -1.28603, -0.713969, -0.206107, -1.25743, -0.742572, -0.235497, -1.22882, -0.771175, -0.264886, -1.20022, -0.799779, -0.294275, -1.17162, -0.828382, -0.323664, -1.14302, -0.856985, -0.353054, -1.11441, -0.885588, -0.382443, -1.08581, -0.914191, -0.411832, -1.05721, -0.942794, -0.441221, -1.0286, -0.971397, -0.470611, -1, -1, -0.5, -0.971397, -1.0286, -0.529389, -0.942794, -1.05721, -0.558779, -0.914191, -1.08581, -0.588168, -0.885588, -1.11441, -0.617557, -0.856985, -1.14302, -0.646946, -0.828382, -1.17162, -0.676336, -0.799779, -1.20022, -0.705725, -0.771175, -1.22882, -0.735114, -0.742572, -1.25743, -0.764503, -0.713969, -1.28603, -0.793893, 0.286031, -0.286031, -0.206107, 0.257428, -0.257428, -0.235497, 0.228825, -0.228825, -0.264886, 0.200221, -0.200221, -0.294275, 0.171618, -0.171618, -0.323664, 0.143015, -0.143015, -0.353054, 0.114412, -0.114412, -0.382443, 0.0858092, -0.0858092, -0.411832, 0.0572061, -0.0572061, -0.441221, 0.0286031, -0.0286031, -0.470611, 0, 0, -0.5, -0.0286031, 0.0286031, -0.529389, -0.0572061, 0.0572061, -0.558779, -0.0858092, 0.0858092, -0.588168, -0.114412, 0.114412, -0.617557, -0.143015, 0.143015, -0.646946, -0.171618, 0.171618, -0.676336, -0.200221, 0.200221, -0.705725, -0.228825, 0.228825, -0.735114, -0.257428, 0.257428, -0.764503, -0.286031, 0.286031, -0.793893, -0, -0.75, -0, -0, -0.75, -0.0375, -0, -0.75, -0.075, -0, -0.75, -0.1125, -0, -0.75, -0.15, -0, -0.75, -0.1875, -0, -0.75, -0.225, -0, -0.75, -0.2625, -0, -0.75, -0.3, -0, -0.75, -0.3375, -0, -0.75, -0.375, -0, -0.75, -0.4125, -0, -0.75, -0.45, -0, -0.75, -0.4875, -0, -0.75, -0.525, -0, -0.75, -0.5625, -0, -0.75, -0.6, -0, -0.75, -0.6375, -0, -0.75, -0.675, -0, -0.75, -0.7125, -0, -0.75, -0.75, -0.75, -0, -0, -0.75, -0, -0.0375, -0.75, -0, -0.075, -0.75, -0, -0.1125, -0.75, -0, -0.15, -0.75, -0, -0.1875, -0.75, -0, -0.225, -0.75, -0, -0.2625, -0.75, -0, -0.3, -0.75, -0, -0.3375, -0.75, -0, -0.375, -0.75, -0, -0.4125, -0.75, -0, -0.45, -0.75, -0, -0.4875, -0.75, -0, -0.525, -0.75, -0, -0.5625, -0.75, -0, -0.6, -0.75, -0, -0.6375, -0.75, -0, -0.675, -0.75, -0, -0.7125, -0.75, -0, -0.75;
    // bp.resize(84);
    // for (int i = 0; i < bbp.rows(); ++i)
    // {
    //     bp[i] = bbp.row(i);
    // }

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



    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", V, F);
    polyscope::state::userCallback = myCallback;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(x);
    polyscope::show();
    return 0;
}
