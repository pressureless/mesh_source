#include <climits>
#include <iostream>
#include <Eigen/Dense>
#include "TriangleMesh.h" 
#include "FaceMesh.h" 
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include "MeshHelper.h"
#include "heartlib.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <ctime>
#include <chrono>
#include <filesystem>
namespace fs = std::filesystem;
inline fs::path DATA_PATH = fs::path(DATA_PATH_STR);

using namespace iheartmesh;
int start;
void print_distance(std::vector<double>& distance){
    std::cout<<"current distance:"<<std::endl;
    for (int m = 0; m < distance.size(); ++m)
    {
        std::cout<<distance[m]<<",";
    } 
}

int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    igl::readOBJ(argc>1?argv[1]: DATA_PATH / "small_bunny.obj", meshV, meshF);
    // igl::readOBJ(argc>1?argv[1]: DATA_PATH / "dragon.obj", meshV, meshF);
    // orient the dragon
    // double angle = 1.57;
    // Eigen::Matrix<double, 3, 3> rotation, rotation1;
    // rotation << std::cos(angle), -std::sin(angle), 0,
    //             std::sin(angle), std::cos(angle), 0,
    //             0, 0, 1;
    // rotation1 << 1,0,0,
    //            0, std::cos(-angle), -std::sin(-angle), 
    //             0 ,std::sin(-angle), std::cos(-angle);
    // for (int i = 0; i < meshV.rows(); ++i)
    // {
    //     meshV.row(i) = rotation1*rotation*meshV.row(i).transpose();
    // }
    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshF);
    FaceMesh face_mesh(triangle_mesh.bm1, triangle_mesh.bm2);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("Geodesic", meshV, meshF);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    heartlib ihla(face_mesh, P);
    std::vector<double > distance;
    std::vector<Eigen::Matrix<double, 3, 1>> N;
    for (int i = 0; i < meshV.rows(); ++i)
    { 
        distance.push_back(10000); 
    } 
 
    int cur = 316;   // small_bunny.obj
    // cur = 4421;   // dragon.obj
    std::vector<std::vector<int> > U;
    std::vector<int> origin;
    origin.push_back(cur);
    distance[cur] = 0;

    std::vector<int> next = origin;
    do
    {
        U.push_back(next);
        next = ihla.GetNextLevel(U);
    } while (next.size() != 0);
    start = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    int i=1, j=1, k=1;
    int max_iter = 2 * U.size();
    while(i <= j){
        std::cout<<"i: "<<i<<", j: "<<j<<", k: "<<k<<std::endl;
        std::vector<double > new_distance;
        for (int index = 0; index < distance.size(); ++index)
        {
            new_distance.push_back(distance[index]);
        }
        std::vector<int> v_set = ihla.GetRangeLevel(U, i, j);
        #pragma omp parallel for schedule(static) num_threads(omp_get_thread_num())
        for (int v: v_set)
        {
            std::vector<int> f_set = ihla.Faces_0(v);
            for (int f: f_set)
            {
                std::tuple< int, int > v_tuple = ihla._Neighborhoods.NeighborVerticesInFace(f, v);
                int v1 = std::get<0>(v_tuple);
                int v2 = std::get<1>(v_tuple);
                double updated = ihla.UpdateStep(v, v1, v2, distance);
                new_distance[v] = std::min(new_distance[v], updated);
            }
        }
        //
        bool all_satisfied = true;
        for (int i = 0; i < distance.size(); ++i)
        {
            double error = std::abs(new_distance[i] - distance[i]) / distance[i];
            if (error > 1e-3)
            {
                all_satisfied = false;
                break;
            }
        }
        if (all_satisfied)
        {
            i++;
        }
        k++;
        if( k < U.size()){
            j = k;
        }
        distance = new_distance;
    } 
    // auto end = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // std::cout <<end-start<< " seconds"<<std::endl;
    // std::cout<<"end: "<<std::endl;
    polyscope::getSurfaceMesh("Geodesic")->addVertexDistanceQuantity("Distance", distance); 
    polyscope::show();
    return 0;
}
