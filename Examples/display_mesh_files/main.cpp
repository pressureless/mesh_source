//
//  main.cpp
//  DEC
//
//  Created by pressure on 10/31/22.
//

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "TriangleMesh.h" 
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/readOFF.h>
#include "MeshHelper.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"


int current_index = 0;
int MAX = 299;
std::string original_file = "/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/cow.off";
std::string file_name = "/Users/pressure/Downloads/ffff/files/new_";

//
Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
bool reading = false;
void renderFile(int index){
    std::string current_file = file_name + std::to_string(index) + ".obj";
    igl::readOBJ(current_file, meshV, meshF);
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(meshV);
}

void initMesh(){
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/cow.off", meshV, meshF);
    igl::readOFF(original_file, meshV, meshF);
    for (int i = 0; i < meshV.rows(); ++i)
    {
        // meshV(i, 1) += 3.73704;
        meshV(i, 1) += 3.638;
    }
}

void myCallback()
{ 
    if (ImGui::Button("Read files")){ 
        reading = true;
    }  
    if (ImGui::Button("Reset")){ 
        reading = false;
        initMesh();
        current_index = 0;
        polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(meshV);
    }  
    if (reading)
    {
        if (current_index<=MAX)
        {
            renderFile(current_index);
            current_index++;  
        }
    }
} 

int main(int argc, const char * argv[]) {
    initMesh();
    // Initialize triangle mesh
    // Initialize polyscope
    polyscope::options::autocenterStructures = false;
    polyscope::options::autoscaleStructures = false;
    polyscope::init();  
    polyscope::options::automaticallyComputeSceneExtents = false;
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::state::userCallback = myCallback; 
    polyscope::show();
    return 0;
}
