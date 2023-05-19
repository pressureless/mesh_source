#include <iostream>
#include <Eigen/Dense>
#include "TriangleMesh.h" 
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/readOFF.h>
#include <igl/readSTL.h>
#include <igl/readMESH.h>
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
Eigen::MatrixXi T;
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
        Velocity[i] = Eigen::Matrix<double, 3, 1>::Zero();
        Force[i] = Eigen::Matrix<double, 3, 1>::Zero();
    } 
    // while(true){
        int TIMES = 25;
        for (int i = 0; i < TIMES; ++i)
        {
            #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
            for (int i = 0; i < meshV.rows(); ++i)
            {
                std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > tuple = ihla.ApplyForces(i, Velocity, Force, Position);
                Velocity[i] = std::get<0>(tuple);
                Position[i] = std::get<1>(tuple);
            } 
            
            #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
            for (int i = 0; i < meshV.rows(); ++i)
            {
                std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > tuple = ihla.ComputeInternalForces(i, Velocity, Position);
                Velocity[i] = std::get<0>(tuple);
                Force[i] = std::get<1>(tuple);
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
        std::cout<<"After updating, min offset: "<<min_diff<<", max offset: "<<max_diff<<std::endl;
        polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(Position);
    // }
}


void myCallback()
{ 
    if (ImGui::Button("Start/stop simulation")){ 
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
    // igl::readOBJ(argc>1?argv[1]:DATA_PATH / "camelhead-decimate-qslim.obj", meshV, meshF); 
    // igl::readOFF(argc>1?argv[1]:DATA_PATH /"bunny_200.off", meshV, meshF);
    igl::readMESH(argc>1?argv[1]:DATA_PATH / "bunny_200.mesh", meshV, T, meshF);   
    double angle = 0.4;
    Eigen::Matrix<double, 3, 3> rotation;
    rotation << std::cos(angle), -std::sin(angle), 0,
                std::sin(angle), std::cos(angle), 0,
                0, 0, 1;
    Eigen::Matrix<double, 3, 1> translation;
    translation << 0, 0, 0; 
    for (int i = 0; i < meshV.rows(); ++i)
    {
        meshV.row(i) = rotation * meshV.row(i).transpose() + translation;
    } 

    // the lowest point is 0.3
    double minY = 1000;
    double offset = 0;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        if (meshV(i, 1) < minY)
        {
            minY = meshV(i, 1);
        }
    } 
    offset = 0.03 - minY;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        meshV(i, 1) += offset;
    }
    // Initialize triangle mesh
    triangle_mesh.initialize(meshF); 
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Position.push_back(meshV.row(i).transpose());
        OriginalPosition.push_back(meshV.row(i).transpose());
        Velocity.push_back(Eigen::Matrix<double, 3, 1>::Zero());
        Force.push_back(Eigen::Matrix<double, 3, 1>::Zero());
    } 
    // Initialize polyscope 
    polyscope::init();   
    polyscope::registerSurfaceMesh("my mesh", OriginalPosition, meshF);
    // polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
    // polyscope::options::groundPlaneHeightFactor = -polyscope::state::lengthScale * 1e-4 + std::get<0>(polyscope::state::boundingBox)[1]; // adjust the plane height
    // polyscope::state::lengthScale = 0;
    polyscope::state::userCallback = myCallback;
    polyscope::show();
    return 0;
}
