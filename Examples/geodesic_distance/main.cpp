//
//  main.cpp
//  DEC
//
//  Created by pressure on 10/31/22.
//
#include <climits>
#include <iostream>
#include <Eigen/Dense>
#include "TriangleMesh.h" 
#include <Eigen/Dense>
#include <Eigen/Sparse> 
#include <igl/readOBJ.h>
#include "MeshHelper.h"
#include "iheartmesh.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"


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
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    igl::readOBJ("../../../models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere3.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/yog.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshF);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("Geodesic", meshV, meshF);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    iheartmesh ihla(triangle_mesh, P);
    std::vector<double > distance;
    std::vector<Eigen::Matrix<double, 3, 1>> N;
    for (int i = 0; i < meshV.rows(); ++i)
    { 
        distance.push_back(10000); 
    } 
 
    int cur = 316;
    // cur = 0;
    std::vector<std::vector<int> > U;
    std::vector<int> origin;
    origin.push_back(cur);
    distance[cur] = 0;

    std::vector<int> next = origin;
    do
    {
        U.push_back(next);
        next = ihla.GetNextLevel(U);
        // next = new std::vector<int>();
        // for (int i = 0; i < ret.size(); ++i)
        // {
        //     next->push_back(ret[i]);
        // }

    } while (next.size() != 0);

    for (int i = 0; i < U.size(); ++i)
    {
        // std::cout<<"current i: "<<i<<std::endl;
        // print_set(U[i]);
    }
    // std::set<int> ran = ihla.GetRangeLevel(U, 1, 2);
    // std::cout<<"ranged: "<<std::endl;
    // print_set(ran);
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
        // std::cout<<"current v_set: "<<std::endl;
        // print_set(v_set);
        for (int v: v_set)
        {
            std::vector<int> f_set = ihla.Faces_0(v);
            for (int f: f_set)
            {
                std::tuple< int, int > v_tuple = ihla._Neighborhoods.NeighborVerticesInFace(f, v);
                int v1 = std::get<0>(v_tuple);
                int v2 = std::get<1>(v_tuple);
                // std::cout<<"f: "<<f<<", v:("<<v<<", "<<v1<<", "<<v2<<")"<<std::endl;
                double updated = ihla.UpdateStep(v, v1, v2, distance);
                // std::cout<<"updated: "<<updated<<std::endl;
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
    std::cout<<"end: "<<std::endl;
    polyscope::getSurfaceMesh("Geodesic")->addVertexDistanceQuantity("Distance", distance); 
    polyscope::show();
    return 0;
}
