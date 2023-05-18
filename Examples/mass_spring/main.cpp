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
#include <igl/readSTL.h>
// #include <thread> 
#include "MeshHelper.h"
#include "heartlib.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

using namespace iheartmesh;

double bottom_z = 0;

Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
Eigen::MatrixXd meshN;
TriangleMesh triangle_mesh;

bool running = false;
double mass = 1.0;
double stiffness = 5e5;
double damping = 5;
double dt = 2e-4;
double eps = 1e-6;

std::vector<Eigen::Matrix<double, 3, 1>> OriginalPosition;
std::vector<Eigen::Matrix<double, 3, 1>> Position;
std::vector<Eigen::Matrix<double, 3, 1>> Velocity;
std::vector<Eigen::Matrix<double, 3, 1>> Force;
 
void update(){
    heartlib ihla(triangle_mesh, OriginalPosition, mass, damping, stiffness, dt, bottom_z);
    for (int i = 0; i < meshV.rows(); ++i)
    {
        //
        Velocity[i] = Eigen::Matrix<double, 3, 1>::Zero();
        Force[i] = Eigen::Matrix<double, 3, 1>::Zero();
    } 
    // while(true){
        int TIMES = 25;
        for (int i = 0; i < TIMES; ++i)
        {
            for (int i = 0; i < meshV.rows(); ++i)
            {
                std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > tuple = ihla.applyForces(i, Velocity, Force, Position);
                Eigen::Matrix<double, 3, 1> vn = std::get<0>(tuple);
                Eigen::Matrix<double, 3, 1> xn = std::get<1>(tuple);
                //
                Velocity[i] = vn;
                Position[i] = xn; 
                // std::cout<<"i:"<<i<<", pos:( "<<xn[0]<<", "<<xn[1]<<", "<<xn[2]<<" )"<<std::endl;
            } 
            
            for (int i = 0; i < meshV.rows(); ++i)
            {
                std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > tuple = ihla.computeInternalForces(i, Velocity, Position);
                Eigen::Matrix<double, 3, 1> vn = std::get<0>(tuple);
                Eigen::Matrix<double, 3, 1> f = std::get<1>(tuple);
                //
                Velocity[i] = vn;
                Force[i] = f; 
            } 
        } 
        
        double min_diff = 1000;
        double max_diff = 0;
        for (int i = 0; i < meshV.rows(); ++i)
        {
            double norm = (Position[i]-OriginalPosition[i]).norm();
            if (norm < min_diff)
            {
                min_diff = norm;
            }
            if(norm > max_diff){
                max_diff = norm;
            } 
        } 
        std::cout<<"After updating, min_offset: "<<min_diff<<", max_offset: "<<max_diff<<std::endl;
    
        polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(Position);
        // polyscope::show();
        // update();
    // }
}


void myCallback()
{ 
    if (ImGui::Button("Start/stop simulation")){ 
        // std::thread first (update); 
        running = !running;
    } 
    if (running)
    {
        update();
    }
}

 

int main(int argc, const char * argv[]) {
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere3.obj", meshV, meshF);
    igl::readOBJ(argc>1?argv[1]:DATA_PATH / "small_disk.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/libigl-polyscope-project/input/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/ddg-exercises/input/sphere.obj", meshV, meshF);
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/bumpy.off", meshV, meshF); // 69KB 5mins
    // igl::readSTL("/Users/pressure/Downloads/small_mesh_from_Thingi10k/1772593.stl", meshV, meshF, meshN); 
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/cow.off", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    double minY = 1000;
    double offset = 0;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        if (meshV(i, 1) < minY)
        {
            minY = meshV(i, 1);
        }
        // std::cout<<"i: "<<i<<", ("<<meshV(i, 0)<<", "<<meshV(i, 1)<<", "<<meshV(i, 2)<<")"<<std::endl;
    } 
    if (minY < 0)
    {
        offset = -minY + 0.1;
    }
    else{
        offset = 0;
    }
    std::cout<<"offset: "<<offset<<std::endl;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        meshV(i, 1) += offset;
    }
    triangle_mesh.initialize(meshF);
    // Initialize polyscope
    polyscope::options::autocenterStructures = false;
    polyscope::options::autoscaleStructures = false;
    polyscope::init();  
    polyscope::options::automaticallyComputeSceneExtents = false;
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::state::userCallback = myCallback;
    Eigen::Matrix<double, 3, 1> initV;
    initV << 0, 0, -100;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Position.push_back(meshV.row(i).transpose());
        OriginalPosition.push_back(meshV.row(i).transpose());
        //
        Velocity.push_back(Eigen::Matrix<double, 3, 1>::Zero());
        Force.push_back(Eigen::Matrix<double, 3, 1>::Zero());
    } 
    polyscope::show();
    return 0;
}
