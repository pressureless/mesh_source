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
#include "MeshHelper.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

using Eigen::MatrixXi;
using Eigen::MatrixXd;

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include "TriangleMesh.h"

struct iheartla {

    std::vector<Eigen::Matrix<double, 3, 1>> x;
    Eigen::Matrix<double, 3, 1> getVertexNormal(
        const int & i)
    {
        Eigen::MatrixXd sum_0 = Eigen::MatrixXd::Zero(3, 1);
        for(int f : Faces(i)){
                // j, k = getNeighborVerticesInFace(f, i)
            std::tuple< int, int > tuple = getNeighborVerticesInFace(f, i);
            int j = std::get<0>(tuple);
            int k = std::get<1>(tuple);
                // n = (x_j- x_i)×(x_k-x_i)
            Eigen::Matrix<double, 3, 1> n = ((x.at(j) - x.at(i))).cross((x.at(k) - x.at(i)));
            sum_0 += n / double((pow((x.at(j) - x.at(i)).lpNorm<2>(), 2) * pow((x.at(k) - x.at(i)).lpNorm<2>(), 2)));
        }
        return (sum_0);    
    }
    struct FundamentalMeshAccessors {
        std::set<int > V;
        std::set<int > E;
        std::set<int > F;
        Eigen::SparseMatrix<int> ve;
        Eigen::SparseMatrix<int> ef;
        Eigen::SparseMatrix<int> uve;
        Eigen::SparseMatrix<int> uef;
        TriangleMesh M;
        std::set<int > Vertices(
            const std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > & S)
        {
            return std::get<1-1>(S);    
        }
        std::set<int > Vertices_0(
            const int & f)
        {
            std::set<int > Vertices_0set_0({f});
            return nonzeros(uve * uef * M.faces_to_vector(Vertices_0set_0));    
        }
        std::set<int > Vertices_1(
            const std::set<int > & f)
        {
            return nonzeros(uve * uef * M.faces_to_vector(f));    
        }
        std::set<int > Vertices_2(
            const int & e)
        {
            std::set<int > Vertices_2set_0({e});
            return nonzeros(uve * M.edges_to_vector(Vertices_2set_0));    
        }
        std::set<int > Vertices_3(
            const std::set<int > & e)
        {
            return nonzeros(uve * M.edges_to_vector(e));    
        }
        std::set<int > Vertices_4(
            const int & v)
        {
            std::set<int > Vertices_4set_0({v});
            return nonzeros(uve * uve.transpose() * M.vertices_to_vector(Vertices_4set_0));    
        }
        std::set<int > Vertices_5(
            const std::set<int > & v)
        {
            return nonzeros(uve * uve.transpose() * M.vertices_to_vector(v));    
        }
        std::set<int > Faces(
            const int & v)
        {
            std::set<int > Facesset_0({v});
            return nonzeros((uve * uef).transpose() * M.vertices_to_vector(Facesset_0));    
        }
        std::set<int > Faces_0(
            const int & e)
        {
            std::set<int > Faces_0set_0({e});
            return nonzeros(uef.transpose() * M.edges_to_vector(Faces_0set_0));    
        }
        std::set<int > Faces_1(
            const std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > & S)
        {
            return std::get<3-1>(S);    
        }
        std::set<int > Edges(
            const std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > & S)
        {
            return std::get<2-1>(S);    
        }
        int Edges(
            const int & i,
            const int & j)
        {
            std::set<int > Edges_0set_0({i});
            std::set<int > Edges_0set_1({j});
            std::set<int > intsect;
            std::set_intersection(nonzeros(ve.transpose() * M.vertices_to_vector(Edges_0set_0)).begin(), nonzeros(ve.transpose() * M.vertices_to_vector(Edges_0set_0)).end(), nonzeros(ve.transpose() * M.vertices_to_vector(Edges_0set_1)).begin(), nonzeros(ve.transpose() * M.vertices_to_vector(Edges_0set_1)).end(), std::inserter(intsect, intsect.begin()));
            std::vector<int> stdv(intsect.begin(), intsect.end());
            Eigen::VectorXi vec(Eigen::Map<Eigen::VectorXi>(&stdv[0], stdv.size()));
            // evec = vec(edgeset(NonZeros(veᵀ IndicatorVector(M, {i}))) ∩ vertexset(NonZeros(veᵀ IndicatorVector(M, {j}))))
            Eigen::VectorXi evec = vec;
            return evec[1-1];    
        }
        std::tuple< int, int > getNeighborVerticesInFace(
            const int & f,
            const int & v)
        {
            std::set<int > getNeighborVerticesInFaceset_0({f});
            // es = edgeset(NonZeros(ef IndicatorVector(M, {f})))
            std::set<int > es = nonzeros(ef * M.faces_to_vector(getNeighborVerticesInFaceset_0));
            std::set<int > getNeighborVerticesInFaceset_1;
            for(int s : es){
                if(ve.coeff(v, s) != 0){
                    getNeighborVerticesInFaceset_1.insert(s);
                }
            }
            // nes = { s for s ∈ es if ve_v,s != 0 }
            std::set<int > nes = getNeighborVerticesInFaceset_1;
            std::set<int > getNeighborVerticesInFaceset_2;
            for(int e : nes){
                if(ef.coeff(e, f) * ve.coeff(v, e) == -1){
                    getNeighborVerticesInFaceset_2.insert(e);
                }
            }
            // eset1 = { e for e ∈ nes if ef_e,f ve_v,e == -1}
            std::set<int > eset1 = getNeighborVerticesInFaceset_2;
            // vset1 = vertexset(NonZeros(uve IndicatorVector(M, eset1)))
            std::set<int > vset1 = nonzeros(uve * M.edges_to_vector(eset1));
            std::set<int > getNeighborVerticesInFaceset_3({v});
            std::set<int > difference;
            std::set_difference(vset1.begin(), vset1.end(), getNeighborVerticesInFaceset_3.begin(), getNeighborVerticesInFaceset_3.end(), std::inserter(difference, difference.begin()));
            std::vector<int> stdv_1(difference.begin(), difference.end());
            Eigen::VectorXi vec_1(Eigen::Map<Eigen::VectorXi>(&stdv_1[0], stdv_1.size()));
            // vvec1 = vec(vset1 - {v})
            Eigen::VectorXi vvec1 = vec_1;
            std::set<int > getNeighborVerticesInFaceset_4;
            for(int e : nes){
                if(ef.coeff(e, f) * ve.coeff(v, e) == 1){
                    getNeighborVerticesInFaceset_4.insert(e);
                }
            }
            // eset2 = { e for e ∈ nes if ef_e,f ve_v,e == 1 }
            std::set<int > eset2 = getNeighborVerticesInFaceset_4;
            // vset2 = vertexset(NonZeros(uve IndicatorVector(M, eset2)))
            std::set<int > vset2 = nonzeros(uve * M.edges_to_vector(eset2));
            std::set<int > getNeighborVerticesInFaceset_5({v});
            std::set<int > difference_1;
            std::set_difference(vset2.begin(), vset2.end(), getNeighborVerticesInFaceset_5.begin(), getNeighborVerticesInFaceset_5.end(), std::inserter(difference_1, difference_1.begin()));
            std::vector<int> stdv_2(difference_1.begin(), difference_1.end());
            Eigen::VectorXi vec_2(Eigen::Map<Eigen::VectorXi>(&stdv_2[0], stdv_2.size()));
            // vvec2 = vec(vset2 - {v})
            Eigen::VectorXi vvec2 = vec_2;
            return std::tuple<int,int >{ vvec1[1-1],vvec2[1-1] };    
        }
        std::tuple< int, int, int > getOrientedVertices(
            const int & f)
        {
            // vs = Vertices(f)
            std::set<int > vs = Vertices_0(f);
            std::vector<int> stdv_3(vs.begin(), vs.end());
            Eigen::VectorXi vec_3(Eigen::Map<Eigen::VectorXi>(&stdv_3[0], stdv_3.size()));
            // vvec = vec(vs)
            Eigen::VectorXi vvec = vec_3;
            // i,j = getNeighborVerticesInFace(f, vvec_1)
            std::tuple< int, int > tuple_3 = getNeighborVerticesInFace(f, vvec[1-1]);
            int i = std::get<0>(tuple_3);
            int j = std::get<1>(tuple_3);
            return std::tuple<int,int,int >{ vvec[1-1],i,j };    
        }
        std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > diamond(
            const int & e)
        {
            std::set<int > diamondset_0({e});
            std::set<int > ss;
            return std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > >{ Vertices_2(e),diamondset_0,Faces_0(e), ss };    
        }
        std::tuple< int, int > oppositeVertices(
            const int & e)
        {
            std::vector<int> stdv_4(Vertices(diamond(e)).begin(), Vertices(diamond(e)).end());
            Eigen::VectorXi vec_4(Eigen::Map<Eigen::VectorXi>(&stdv_4[0], stdv_4.size()));
            // evec = vec(Vertices(diamond(e)))
            Eigen::VectorXi evec = vec_4;
            return std::tuple<int,int >{ evec[1-1],evec[2-1] };    
        }
        FundamentalMeshAccessors(const TriangleMesh & M)
        {
            // V, E, F = MeshSets( M )
            std::tuple< std::set<int >, std::set<int >, std::set<int > > tuple = M.MeshSets();
            V = std::get<0>(tuple);
            E = std::get<1>(tuple);
            F = std::get<2>(tuple);
            int dimv_0 = M.n_vertices();
            int dime_0 = M.n_edges();
            int dimf_0 = M.n_faces();
            this->M = M;
            // ve, ef = BoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_1 = M.BoundaryMatrices();
            ve = std::get<0>(tuple_1);
            ef = std::get<1>(tuple_1);
            // uve, uef = UnsignedBoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_2 = M.UnsignedBoundaryMatrices();
            uve = std::get<0>(tuple_2);
            uef = std::get<1>(tuple_2);
        }
    };
    FundamentalMeshAccessors _FundamentalMeshAccessors;
    std::tuple< int, int > getNeighborVerticesInFace(int p0,int p1){
        return _FundamentalMeshAccessors.getNeighborVerticesInFace(p0,p1);
    };
    std::set<int > Faces(int p0){
        return _FundamentalMeshAccessors.Faces(p0);
    };
    std::set<int > Faces_0(int p0){
        return _FundamentalMeshAccessors.Faces_0(p0);
    };
    std::set<int > Faces_1(std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > p0){
        return _FundamentalMeshAccessors.Faces_1(p0);
    };
    std::set<int > Vertices(std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > p0){
        return _FundamentalMeshAccessors.Vertices(p0);
    };
    std::set<int > Vertices_0(int p0){
        return _FundamentalMeshAccessors.Vertices_0(p0);
    };
    std::set<int > Vertices_1(std::set<int > p0){
        return _FundamentalMeshAccessors.Vertices_1(p0);
    };
    std::set<int > Vertices_2(int p0){
        return _FundamentalMeshAccessors.Vertices_2(p0);
    };
    std::set<int > Vertices_3(std::set<int > p0){
        return _FundamentalMeshAccessors.Vertices_3(p0);
    };
    std::set<int > Vertices_4(int p0){
        return _FundamentalMeshAccessors.Vertices_4(p0);
    };
    std::set<int > Vertices_5(std::set<int > p0){
        return _FundamentalMeshAccessors.Vertices_5(p0);
    };
    iheartla(
        const TriangleMesh & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    :
    _FundamentalMeshAccessors(M)
    {
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        int dimf_0 = M.n_faces();
        const long dim_0 = x.size();
        this->x = x;
    
    }
};



int main(int argc, const char * argv[]) {
    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
    igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);

    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshV, meshF);

    // Initialize polyscope
    polyscope::init(); 

    // Register a surface mesh structure
    // `meshVerts` is a Vx3 array-like container of vertex positions
    // `meshFaces` is a Fx3 array-like container of face indices  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    std::vector<Eigen::Matrix<double, 3, 1>> P;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    iheartla ihla(triangle_mesh, P);
    std::vector<Eigen::Matrix<double, 3, 1>> N;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        N.push_back(ihla.getVertexNormal(i));
        std::cout<<"ihla.getVertexNormal(i):\n"<<ihla.getVertexNormal(i)<<std::endl;
    }

    // Add a scalar and a vector function defined on the mesh
    // `scalarQuantity` is a length V array-like container of values
    // `vectorQuantity` is an Fx3 array-like container of vectors per face
    // polyscope::getSurfaceMesh("my mesh")->addVertexScalarQuantity("my_scalar", scalarQuantity);
    // polyscope::getSurfaceMesh("my mesh")->addFaceVectorQuantity("my_vector", vectorQuantity);

    // View the point cloud and mesh we just registered in the 3D UI
    polyscope::show();

    return 0;
}
