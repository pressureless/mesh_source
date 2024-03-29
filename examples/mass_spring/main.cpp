#include <iostream>
#include <Eigen/Dense>
#include "Tetrahedron.h"
#include "CellMesh.h"
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
Tetrahedron tet_mesh;

bool running = false;
double mass = 1.0;
double stiffness = 5e5;
double damping = 5;
// double damping = 15;
double dt = 2e-4;
double eps = 1e-6;

std::vector<Eigen::Matrix<double, 3, 1>> OriginalPosition;
std::vector<Eigen::Matrix<double, 3, 1>> Position;
std::vector<Eigen::Matrix<double, 3, 1>> Velocity;
std::vector<Eigen::Matrix<double, 3, 1>> Force;
heartlib *_heartlib;
void update(){
    int TIMES = 25;
    for (int i = 0; i < TIMES; ++i)
    {
        #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
        for (int i = 0; i < meshV.rows(); ++i)
        {
            std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > tuple = _heartlib->ApplyForces(i, Velocity, Force, Position);
            Velocity[i] = std::get<0>(tuple);
            Position[i] = std::get<1>(tuple);
        } 
        
        #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
        for (int i = 0; i < meshV.rows(); ++i)
        {
            std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > tuple = _heartlib->ComputeInternalForces(i, Velocity, Position);
            Velocity[i] = std::get<0>(tuple);
            Force[i] = std::get<1>(tuple);
        } 
    } 
    // double min_diff = 1000;
    // double max_diff = 0;
    // for (int i = 0; i < meshV.rows(); ++i)
    // {
    //     double norm = (Position[i]-OriginalPosition[i]).norm();
    //     if (norm < min_diff)
    //     {
    //         min_diff = norm;
    //     }
    //     if(norm > max_diff){
    //         max_diff = norm;
    //     } 
    // } 
    // std::cout<<"After updating, min offset: "<<min_diff<<", max offset: "<<max_diff<<std::endl;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(Position);
}

void reset(){
    Eigen::Matrix<double, 3, 1> initV;
    initV<< -10, 0, 0;
    // initV<< -15, 0, 0;
    Eigen::Matrix<double, 3, 1> initF;
    initF<< -1000, -50000, 0;
    // initF<< -14000, -100000, 0;
    #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Position[i] = OriginalPosition[i]; 
        Velocity[i] = initV.transpose(); 
        Force[i] = initF.transpose(); 
    }  
}


void myCallback()
{ 
    if (ImGui::Button("Start/stop simulation")){ 
        running = !running;
    } 
    if (ImGui::Button("Reset")){ 
        reset();
    } 
    if (running)
    {
        update();
    }
}

 

int main(int argc, const char * argv[]) {
    igl::readMESH(argc>1?argv[1]:DATA_PATH / "bunny_200.mesh", meshV, T, meshF);   
    // igl::readMESH(argc>1?argv[1]:DATA_PATH / "bunny1k.mesh", meshV, T, meshF);   
    // double angle = 0.4;
    // Eigen::Matrix<double, 3, 3> rotation;
    // rotation << std::cos(angle), -std::sin(angle), 0,
    //             std::sin(angle), std::cos(angle), 0,
    //             0, 0, 1;
    // Eigen::Matrix<double, 3, 1> translation;
    // translation << 0, 0, 0; 
    // for (int i = 0; i < meshV.rows(); ++i)
    // {
    //     meshV.row(i) = rotation * meshV.row(i).transpose() + translation;
    // } 
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
    offset = 0.3+0.03 - minY;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        meshV(i, 1) += offset;
    }
    // Initialize tet mesh
    tet_mesh.initialize(T); 
    for (int i = 0; i < meshV.rows(); ++i)
    {
        OriginalPosition.push_back(meshV.row(i).transpose());
    } 
    CellMesh cell_mesh(tet_mesh.bm1, tet_mesh.bm2, tet_mesh.bm3);
    _heartlib = new heartlib(cell_mesh, OriginalPosition, mass, damping, stiffness, dt, bottom_z);
    Position.resize(OriginalPosition.size());
    Velocity.resize(OriginalPosition.size());
    Force.resize(OriginalPosition.size());
    reset();
    // Initialize polyscope 
    polyscope::init();   
    polyscope::registerSurfaceMesh("my mesh", OriginalPosition, meshF);
    // polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
    polyscope::options::groundPlaneHeightFactor = 1; // adjust the plane height
    // polyscope::state::lengthScale = 0;
    polyscope::state::userCallback = myCallback;
    polyscope::show();
    return 0;
}
