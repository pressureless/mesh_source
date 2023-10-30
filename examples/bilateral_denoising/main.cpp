#include <iostream>
#include <Eigen/Dense>
#include "TriangleMesh.h" 
#include "FaceMesh.h" 
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
FaceMesh *face_mesh;

std::vector<Eigen::Matrix<double, 3, 1>> P;

void update(){
    heartlib ihla(*face_mesh, P); 
    std::vector<Eigen::Matrix<double, 3, 1>> NP;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Eigen::Matrix<double, 3, 1> new_pos = ihla.DenoisePoint(i);
        NP.push_back(new_pos);
    } 
    P = NP;
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
    igl::readOBJ(DATA_PATH / "torusnoise.obj", meshV, meshF); // 177KB 20 mins
    // Initialize triangle mesh
    triangle_mesh.initialize(meshF);
    face_mesh = new FaceMesh(triangle_mesh.bm1, triangle_mesh.bm2);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::state::userCallback = myCallback;

    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    polyscope::show();
    return 0;
}
