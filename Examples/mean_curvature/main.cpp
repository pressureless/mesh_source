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
#include <igl/writeOBJ.h>
#include <igl/readOFF.h>
#include "MeshHelper.h"
#include "iheartmesh.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
TriangleMesh triangle_mesh;

double cg_tolerance = 1e-6;
int MAX_ITERATION = 100;
double step = 1e-3;

std::vector<Eigen::Matrix<double, 3, 1>> OriginalPosition;
std::vector<Eigen::Matrix<double, 3, 1>> Position;

void axpy3(const std::vector<Eigen::Matrix<double, 3, 1>>& X,
           const double                            alpha,
           const double                            beta,
           std::vector<Eigen::Matrix<double, 3, 1>>&       Y)
{
    // Y = beta*Y + alpha*X 
    int size = static_cast<int>(X.size()); 
    for (int i = 0; i < size; ++i) {
        Y[i] *= beta;
        Y[i] += alpha * X[i];
    }
}

double dot3(const std::vector<Eigen::Matrix<double, 3, 1>>& A,
       const std::vector<Eigen::Matrix<double, 3, 1>>& B)
{

    double  ret = 0;
    int size = static_cast<int>(A.size());
    for (int i = 0; i < size; ++i) {
        ret += A[i].dot(B[i]);
    }
    return ret;
}


void conjugate_gradients_algorithm(std::vector<Eigen::Matrix<double, 3, 1>>& X,
        std::vector<Eigen::Matrix<double, 3, 1>>& B,
        std::vector<Eigen::Matrix<double, 3, 1>>& R,
        std::vector<Eigen::Matrix<double, 3, 1>>& P,
        std::vector<Eigen::Matrix<double, 3, 1>>& S,
        double& start_residual,
        double&  stop_residual,
        iheartmesh& ihla)
{
    // Conjugate Gradients
    // Page 50 in "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"
    // CG solver. Solve for the three coordinates simultaneously

    // s = Ax 
    for (int i = 0; i < meshV.rows(); ++i)
    {
        S[i] = ihla.Ax(i, X);
    } 
 
    // r = b - s = b - Ax
    // p = r 
    for (int i = 0; i < int(meshV.rows()); ++i) {
        R[i] = B[i] - S[i];
        R[i] = B[i] - S[i];
        R[i] = B[i] - S[i];

        P[i] = R[i];
        P[i] = R[i];
        P[i] = R[i];
    }
 
    // delta_new = <r,r>
    double delta_new = dot3(R, R);

    // delta_0 = delta_new
    double delta_0 = delta_new;
        std::cout<<"delta_new delta_new: "<<delta_new<<std::endl;

    start_residual = delta_0;
    uint32_t iter  = 0;
    while (iter < MAX_ITERATION) { 
        // s = Ap 
        for (int i = 0; i < meshV.rows(); ++i)
        {
            S[i] = ihla.Ax(i, P);
        }  

        // alpha = delta_new / <s,p>
        double alpha = dot3(S, P);
        std::cout<<"alpha before: "<<alpha<<std::endl;
        alpha = delta_new / alpha;

        std::cout<<"alpha: "<<alpha<<std::endl;
        // x =  x + alpha*p
        axpy3(P, alpha, 1.0, X);

        // r = r - alpha*s
        axpy3(S, -alpha, 1.0, R);

        // delta_old = delta_new
        double delta_old(delta_new);

        // delta_new = <r,r>
        delta_new = dot3(R, R);

        // beta = delta_new/delta_old
        double beta(delta_new / delta_old);

        // exit if error is getting too low across three coordinates
        std::cout<<"delta_new:"<<delta_new<<std::endl;
        std::cout<<"cg_tolerance * cg_tolerance * delta_0:"<<cg_tolerance * cg_tolerance * delta_0<<std::endl;
            
        if (delta_new < cg_tolerance * cg_tolerance * delta_0) {
            break;
        }

        // p = beta*p + r
        axpy3(R, 1.0, beta, P);

        ++iter;
        std::cout<<"iter:"<<iter<<std::endl;
    }  
    stop_residual = delta_new;
}

void update(){
    iheartmesh ihla(triangle_mesh, step); 
    std::cout<<"After"<<std::endl;
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout<<"i: "<<i<<", "<<P[i]<<std::endl;
    // } 
    double start_residual, stop_residual;
    std::vector<Eigen::Matrix<double, 3, 1>> X(meshV.rows()), B(meshV.rows()), R(meshV.rows()), P(meshV.rows()), S(meshV.rows());

    for (int i = 0; i < meshV.rows(); ++i)
    {
        X[i] = Position[i];
        B[i] = Position[i];
    } 
    conjugate_gradients_algorithm(X, B, R, P, S, start_residual, stop_residual, ihla);
    std::cout<<"start_residual:"<<start_residual<<", stop_residual:"<<stop_residual<<std::endl;
    
    Position = X;
    double min_diff = 1000;
    double max_diff = 0;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        double norm = (X[i]-OriginalPosition[i]).norm();
        if (norm < min_diff)
        {
            min_diff = norm;
        }
        if(norm > max_diff){
            max_diff = norm;
        }
        // std::cout<<"i: "<<i<<", ("<<X[i][0]-OriginalPosition[i][0]<<", "<<X[i][1]-OriginalPosition[i][1]<<", "<<X[i][2]-OriginalPosition[i][2]<<")"<<std::endl;
    } 
    std::cout<<"After updating, min_offset: "<<min_diff<<", max_offset: "<<max_diff<<std::endl;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(Position);
}


void myCallback()
{ 
    if (ImGui::Button("One step")){ 
        update();
    } 
    if (ImGui::Button("Ten steps")){
        for (int i = 0; i < 10; ++i)
        {
            update();
        }
    } 
    if (ImGui::Button("Fifty steps")){
        for (int i = 0; i < 50; ++i)
        {
            update();
        }
    } 
}

 

int main(int argc, const char * argv[]) {
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
    igl::readOBJ("../../../../models/sphere3.obj", meshV, meshF);
    // igl::readOBJ("../../../../models/small_disk.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/libigl-polyscope-project/input/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/ddg-exercises/input/sphere.obj", meshV, meshF);
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/bumpy.off", meshV, meshF); // 69KB 5mins
    
    // igl::readOFF("../../../../models/cow.off", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    triangle_mesh.initialize(meshF);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::state::userCallback = myCallback;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Position.push_back(meshV.row(i).transpose());
        OriginalPosition.push_back(meshV.row(i).transpose());
    } 
    // for (int i = 0; i < meshV.rows(); ++i)
    // {
    //     std::cout<<"i: "<<i<<", ("<<Position[i][0]<<", "<<Position[i][1]<<", "<<Position[i][2]<<")"<<std::endl;
    // } 
    polyscope::show();
    return 0;
}
