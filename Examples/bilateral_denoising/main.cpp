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
#include "heartlib.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

using namespace iheartmesh;

Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
TriangleMesh triangle_mesh;

std::vector<Eigen::Matrix<double, 3, 1>> P;

void update(){
    heartlib ihla(triangle_mesh, P); 
    // std::cout<<"before"<<std::endl;
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout<<"i: "<<i<<", "<<P[i]<<std::endl;
    // } 
    std::vector<Eigen::Matrix<double, 3, 1>> NP;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Eigen::Matrix<double, 3, 1> new_pos = ihla.DenoisePoint(i);
        NP.push_back(new_pos);
    } 
    P = NP;
    // std::cout<<"after"<<std::endl;
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout<<"i: "<<i<<", "<<P[i]<<std::endl;
    // } 
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(P);
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
        update();
    } 
    if (ImGui::Button("Five steps")){
        for (int i = 0; i < 5; ++i)
        {
            update();
        }
    } 
    if (ImGui::Button("Save current positions to file")){
        Eigen::MatrixXd VV(meshV.rows(), 3);
        for (int i = 0; i < VV.rows(); ++i)
        {
            VV.row(i) = P[i];
        }
        igl::writeOBJ(DATA_PATH / "updated_torusnoise.obj", VV, meshF);
    } 
}

int main(int argc, const char * argv[]) {
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/Mesh_Denoiseing_BilateralFilter/Noisy.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/fast-mesh-denoising/meshes/Noisy/block_n1.obj", meshV, meshF);
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/bumpy.off", meshV, meshF); // 69KB 5mins
    igl::readOBJ(DATA_PATH / "torusnoise.obj", meshV, meshF); // 177KB 20 mins
    

    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    triangle_mesh.initialize(meshF);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::state::userCallback = myCallback;

    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    // update();
    // heartlib ihla(triangle_mesh, P); 
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
