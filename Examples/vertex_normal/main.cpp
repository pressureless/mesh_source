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
/*
FaceNeighbors, NeighborVerticesInFace from Neighborhoods(M)
M ∈ mesh
x_i ∈ ℝ^3  

VertexNormal(i) = (sum_(f ∈ FaceNeighbors(i)) (x_j- x_i)×(x_k-x_i)/(||x_j-x_i||² ||x_k-x_i||²) 
where j, k = NeighborVerticesInFace(f, i) ) where i ∈ ℤ vertices
*/
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include <algorithm>

struct iheartla {

    std::vector<Eigen::Matrix<double, 3, 1>> x;
    Eigen::Matrix<double, 3, 1> VertexNormal(
        const int & i)
    {
        Eigen::MatrixXd sum_0 = Eigen::MatrixXd::Zero(3, 1);
        for(int f : FaceNeighbors(i)){
                // j, k = NeighborVerticesInFace(f, i)
            std::tuple< int, int > tuple = NeighborVerticesInFace(f, i);
            int j = std::get<0>(tuple);
            int k = std::get<1>(tuple);
            sum_0 += ((x.at(j) - x.at(i))).cross((x.at(k) - x.at(i))) / double((pow((x.at(j) - x.at(i)).lpNorm<2>(), 2) * pow((x.at(k) - x.at(i)).lpNorm<2>(), 2)));
        }
        return (sum_0);    
    }
    struct Neighborhoods {
        std::set<int > V;
        std::set<int > E;
        std::set<int > F;
        Eigen::SparseMatrix<int> dee1;
        Eigen::SparseMatrix<int> dee2;
        Eigen::SparseMatrix<int> B0;
        Eigen::SparseMatrix<int> B1;
        TriangleMesh M;
        std::set<int > VertexOneRing(
            const int & v)
        {
            assert( V.find(v) != V.end() );
            std::set<int > VertexOneRingset_0({v});
            std::set<int > VertexOneRingset_1({v});
            std::set<int > difference;
            std::set<int > lhs_diff = nonzeros(B0 * B0.transpose() * M.vertices_to_vector(VertexOneRingset_0));
            std::set<int > rhs_diff = VertexOneRingset_1;
            std::set_difference(lhs_diff.begin(), lhs_diff.end(), rhs_diff.begin(), rhs_diff.end(), std::inserter(difference, difference.begin()));
            return difference;    
        }
        std::set<int > VertexOneRing(
            const std::set<int > & v)
        {
            std::set<int > difference_1;
            std::set<int > lhs_diff_1 = nonzeros(B0 * B0.transpose() * M.vertices_to_vector(v));
            std::set<int > rhs_diff_1 = v;
            std::set_difference(lhs_diff_1.begin(), lhs_diff_1.end(), rhs_diff_1.begin(), rhs_diff_1.end(), std::inserter(difference_1, difference_1.begin()));
            return difference_1;    
        }
        std::set<int > FaceNeighbors(
            const int & v)
        {
            assert( V.find(v) != V.end() );
            std::set<int > FaceNeighborsset_0({v});
            return nonzeros((B0 * B1).transpose() * M.vertices_to_vector(FaceNeighborsset_0));    
        }
        std::set<int > FaceNeighbors_0(
            const int & e)
        {
            assert( E.find(e) != E.end() );
            std::set<int > FaceNeighbors_0set_0({e});
            return nonzeros(B1.transpose() * M.edges_to_vector(FaceNeighbors_0set_0));    
        }
        int EdgeIndex(
            const int & i,
            const int & j)
        {
            assert( V.find(j) != V.end() );
            std::set<int > EdgeIndexset_0({i});
            std::set<int > EdgeIndexset_1({j});
            std::set<int > intsect;
            std::set<int > lhs = nonzeros(dee1.transpose() * M.vertices_to_vector(EdgeIndexset_0));
            std::set<int > rhs = nonzeros(dee1.transpose() * M.vertices_to_vector(EdgeIndexset_1));
            std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::inserter(intsect, intsect.begin()));
            std::set<int > op = intsect;
            std::vector<int> stdv(op.begin(), op.end());
            Eigen::VectorXi vec(Eigen::Map<Eigen::VectorXi>(&stdv[0], stdv.size()));
            // evec = vec(edgeset(NonZeros(∂1ᵀ IndicatorVector(M, {i}))) ∩ vertexset(NonZeros(∂1ᵀ IndicatorVector(M, {j}))))
            Eigen::VectorXi evec = vec;
            return evec[1-1];    
        }
        std::tuple< int, int > NeighborVerticesInFace(
            const int & f,
            const int & v)
        {
            assert( F.find(f) != F.end() );
            std::set<int > NeighborVerticesInFaceset_0({f});
            // es = edgeset(NonZeros(∂2 IndicatorVector(M, {f})))
            std::set<int > es = nonzeros(dee2 * M.faces_to_vector(NeighborVerticesInFaceset_0));
            std::set<int > NeighborVerticesInFaceset_1;
            for(int s : es){
                if(dee1.coeff(v, s) != 0){
                    NeighborVerticesInFaceset_1.insert(s);
                }
            }
            // nes = { s for s ∈ es if ∂1_v,s != 0 }
            std::set<int > nes = NeighborVerticesInFaceset_1;
            std::set<int > NeighborVerticesInFaceset_2;
            for(int e : nes){
                if(dee2.coeff(e, f) * dee1.coeff(v, e) == -1){
                    NeighborVerticesInFaceset_2.insert(e);
                }
            }
            // eset1 = { e for e ∈ nes if ∂2_e,f ∂1_v,e == -1}
            std::set<int > eset1 = NeighborVerticesInFaceset_2;
            // vset1 = vertexset(NonZeros(B0 IndicatorVector(M, eset1)))
            std::set<int > vset1 = nonzeros(B0 * M.edges_to_vector(eset1));
            std::set<int > NeighborVerticesInFaceset_3({v});
            std::set<int > difference_2;
            std::set<int > lhs_diff_2 = vset1;
            std::set<int > rhs_diff_2 = NeighborVerticesInFaceset_3;
            std::set_difference(lhs_diff_2.begin(), lhs_diff_2.end(), rhs_diff_2.begin(), rhs_diff_2.end(), std::inserter(difference_2, difference_2.begin()));
            std::set<int > op_1 = difference_2;
            std::vector<int> stdv_1(op_1.begin(), op_1.end());
            Eigen::VectorXi vec_1(Eigen::Map<Eigen::VectorXi>(&stdv_1[0], stdv_1.size()));
            // vvec1 = vec(vset1 - {v})
            Eigen::VectorXi vvec1 = vec_1;
            std::set<int > NeighborVerticesInFaceset_4;
            for(int e : nes){
                if(dee2.coeff(e, f) * dee1.coeff(v, e) == 1){
                    NeighborVerticesInFaceset_4.insert(e);
                }
            }
            // eset2 = { e for e ∈ nes if ∂2_e,f ∂1_v,e == 1 }
            std::set<int > eset2 = NeighborVerticesInFaceset_4;
            // vset2 = vertexset(NonZeros(B0 IndicatorVector(M, eset2)))
            std::set<int > vset2 = nonzeros(B0 * M.edges_to_vector(eset2));
            std::set<int > NeighborVerticesInFaceset_5({v});
            std::set<int > difference_3;
            std::set<int > lhs_diff_3 = vset2;
            std::set<int > rhs_diff_3 = NeighborVerticesInFaceset_5;
            std::set_difference(lhs_diff_3.begin(), lhs_diff_3.end(), rhs_diff_3.begin(), rhs_diff_3.end(), std::inserter(difference_3, difference_3.begin()));
            std::set<int > op_2 = difference_3;
            std::vector<int> stdv_2(op_2.begin(), op_2.end());
            Eigen::VectorXi vec_2(Eigen::Map<Eigen::VectorXi>(&stdv_2[0], stdv_2.size()));
            // vvec2 = vec(vset2 - {v})
            Eigen::VectorXi vvec2 = vec_2;
            return std::tuple<int,int >{ vvec1[1-1],vvec2[1-1] };    
        }
        std::tuple< int, int, int > OrientedVertices(
            const int & f)
        {
            assert( F.find(f) != F.end() );
            // vs = Vertices(f)
            std::set<int > vs = Vertices_0(f);
            std::set<int > op_3 = vs;
            std::vector<int> stdv_3(op_3.begin(), op_3.end());
            Eigen::VectorXi vec_3(Eigen::Map<Eigen::VectorXi>(&stdv_3[0], stdv_3.size()));
            // vvec = vec(vs)
            Eigen::VectorXi vvec = vec_3;
            // i,j = NeighborVerticesInFace(f, vvec_1)
            std::tuple< int, int > tuple_3 = NeighborVerticesInFace(f, vvec[1-1]);
            int i = std::get<0>(tuple_3);
            int j = std::get<1>(tuple_3);
            return std::tuple<int,int,int >{ vvec[1-1],i,j };    
        }
        std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > Diamond(
            const int & e)
        {
            assert( E.find(e) != E.end() );
            std::set<int > Diamondset_0({e});
        std::set<int > tetset;
            return std::tuple<std::set<int >,std::set<int >,std::set<int >,std::set<int > >{ Vertices_2(e),Diamondset_0,FaceNeighbors_0(e),tetset };    
        }
        std::tuple< int, int > OppositeVertices(
            const int & e)
        {
            assert( E.find(e) != E.end() );
            std::set<int > difference_4;
            std::set<int > lhs_diff_4 = Vertices_1(FaceNeighbors_0(e));
            std::set<int > rhs_diff_4 = Vertices_2(e);
            std::set_difference(lhs_diff_4.begin(), lhs_diff_4.end(), rhs_diff_4.begin(), rhs_diff_4.end(), std::inserter(difference_4, difference_4.begin()));
            std::set<int > op_4 = difference_4;
            std::vector<int> stdv_4(op_4.begin(), op_4.end());
            Eigen::VectorXi vec_4(Eigen::Map<Eigen::VectorXi>(&stdv_4[0], stdv_4.size()));
            // evec = vec(Vertices(FaceNeighbors(e)) \ Vertices(e))
            Eigen::VectorXi evec = vec_4;
            return std::tuple<int,int >{ evec[1-1],evec[2-1] };    
        }
        int FaceIndex(
            const int & i,
            const int & j,
            const int & k)
        {
            assert( V.find(k) != V.end() );
            // ufv = (B0 B1)ᵀ
            Eigen::SparseMatrix<int> ufv = (B0 * B1).transpose();
            std::set<int > FaceIndexset_0({i});
            // iface = faceset(NonZeros(ufv  IndicatorVector(M, {i})))
            std::set<int > iface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_0));
            std::set<int > FaceIndexset_1({j});
            // jface = faceset(NonZeros(ufv  IndicatorVector(M, {j})))
            std::set<int > jface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_1));
            std::set<int > FaceIndexset_2({k});
            // kface = faceset(NonZeros(ufv IndicatorVector(M, {k})))
            std::set<int > kface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_2));
            std::set<int > intsect_1;
            std::set<int > lhs_1 = jface;
            std::set<int > rhs_1 = kface;
            std::set_intersection(lhs_1.begin(), lhs_1.end(), rhs_1.begin(), rhs_1.end(), std::inserter(intsect_1, intsect_1.begin()));
            std::set<int > intsect_2;
            std::set<int > lhs_2 = iface;
            std::set<int > rhs_2 = intsect_1;
            std::set_intersection(lhs_2.begin(), lhs_2.end(), rhs_2.begin(), rhs_2.end(), std::inserter(intsect_2, intsect_2.begin()));
            std::set<int > op_5 = intsect_2;
            std::vector<int> stdv_5(op_5.begin(), op_5.end());
            Eigen::VectorXi vec_5(Eigen::Map<Eigen::VectorXi>(&stdv_5[0], stdv_5.size()));
            // fvec = vec(iface ∩ jface ∩ kface)
            Eigen::VectorXi fvec = vec_5;
            return fvec[1-1];    
        }
        std::tuple< int, int > OrientedVertices(
            const int & i,
            const int & j,
            const int & k)
        {
            assert( V.find(k) != V.end() );
            // f = FaceIndex(i, j, k)
            int f = FaceIndex(i, j, k);
            return NeighborVerticesInFace(f, i);    
        }
        struct FundamentalMeshAccessors {
            std::set<int > V;
            std::set<int > E;
            std::set<int > F;
            Eigen::SparseMatrix<int> dee1;
            Eigen::SparseMatrix<int> dee2;
            Eigen::SparseMatrix<int> B0;
            Eigen::SparseMatrix<int> B1;
            TriangleMesh M;
            std::set<int > Vertices(
                const std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > & S)
            {
                return std::get<1-1>(S);    
            }
            std::set<int > Edges(
                const std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > & S)
            {
                return std::get<2-1>(S);    
            }
            std::set<int > Faces(
                const std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > & S)
            {
                return std::get<3-1>(S);    
            }
            std::set<int > Vertices_0(
                const int & f)
            {
                assert( F.find(f) != F.end() );
                std::set<int > Vertices_0set_0({f});
                return nonzeros(B0 * B1 * M.faces_to_vector(Vertices_0set_0));    
            }
            std::set<int > Vertices_1(
                const std::set<int > & G)
            {
                return nonzeros(B0 * B1 * M.faces_to_vector(G));    
            }
            std::set<int > Vertices_2(
                const int & e)
            {
                assert( E.find(e) != E.end() );
                std::set<int > Vertices_2set_0({e});
                return nonzeros(B0 * M.edges_to_vector(Vertices_2set_0));    
            }
            std::set<int > Vertices_3(
                const std::set<int > & H)
            {
                return nonzeros(B0 * M.edges_to_vector(H));    
            }
            std::set<int > Edges_0(
                const int & v)
            {
                assert( V.find(v) != V.end() );
                std::set<int > Edges_0set_0({v});
                return nonzeros(B0.transpose() * M.vertices_to_vector(Edges_0set_0));    
            }
            std::set<int > Edges_1(
                const int & f)
            {
                assert( F.find(f) != F.end() );
                std::set<int > Edges_1set_0({f});
                return nonzeros(B1 * M.faces_to_vector(Edges_1set_0));    
            }
            std::set<int > Faces_0(
                const int & v)
            {
                assert( V.find(v) != V.end() );
                std::set<int > Faces_0set_0({v});
                return nonzeros((B0 * B1).transpose() * M.vertices_to_vector(Faces_0set_0));    
            }
            std::set<int > Faces_1(
                const int & e)
            {
                assert( E.find(e) != E.end() );
                std::set<int > Faces_1set_0({e});
                return nonzeros(B1.transpose() * M.edges_to_vector(Faces_1set_0));    
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
                // dee1, dee2 = BoundaryMatrices(M)
                std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_1 = M.BoundaryMatrices();
                dee1 = std::get<0>(tuple_1);
                dee2 = std::get<1>(tuple_1);
                // B0, B1 = UnsignedBoundaryMatrices(M)
                std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_2 = M.UnsignedBoundaryMatrices();
                B0 = std::get<0>(tuple_2);
                B1 = std::get<1>(tuple_2);
            }
        };
        FundamentalMeshAccessors _FundamentalMeshAccessors;
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
        std::set<int > Edges(std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > p0){
            return _FundamentalMeshAccessors.Edges(p0);
        };
        std::set<int > Edges_0(int p0){
            return _FundamentalMeshAccessors.Edges_0(p0);
        };
        std::set<int > Edges_1(int p0){
            return _FundamentalMeshAccessors.Edges_1(p0);
        };
        std::set<int > Faces(std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > p0){
            return _FundamentalMeshAccessors.Faces(p0);
        };
        std::set<int > Faces_0(int p0){
            return _FundamentalMeshAccessors.Faces_0(p0);
        };
        std::set<int > Faces_1(int p0){
            return _FundamentalMeshAccessors.Faces_1(p0);
        };
        Neighborhoods(const TriangleMesh & M)
        :
        _FundamentalMeshAccessors(M)
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
            // ∂1, ∂2 = BoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_1 = M.BoundaryMatrices();
            dee1 = std::get<0>(tuple_1);
            dee2 = std::get<1>(tuple_1);
            // B0, B1 = UnsignedBoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_2 = M.UnsignedBoundaryMatrices();
            B0 = std::get<0>(tuple_2);
            B1 = std::get<1>(tuple_2);
        }
    };
    Neighborhoods _Neighborhoods;
    std::set<int > FaceNeighbors(int p0){
        return _Neighborhoods.FaceNeighbors(p0);
    };
    std::set<int > FaceNeighbors_0(int p0){
        return _Neighborhoods.FaceNeighbors_0(p0);
    };
    std::tuple< int, int > NeighborVerticesInFace(int p0,int p1){
        return _Neighborhoods.NeighborVerticesInFace(p0,p1);
    };
    iheartla(
        const TriangleMesh & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    :
    _Neighborhoods(M)
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
    igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    TriangleMesh triangle_mesh;
    triangle_mesh.initialize(meshV, meshF);
    // Initialize polyscope
    polyscope::init();  
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
        Eigen::Matrix<double, 3, 1> n = ihla.VertexNormal(i);
        N.push_back(n);
        // std::cout<<"ihla.getVertexNormal(i):\n"<<n<<std::endl;
    } 
    polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("VertexNormal", N); 
    polyscope::show();
    return 0;
}
