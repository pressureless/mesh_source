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
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

double bottom_z = 0;


/*
VertexOneRing from Neighborhoods(M)
M ∈ mesh
x_i ∈ ℝ^3: original positions
m ∈ ℝ: mass
damping ∈ ℝ: damping
K ∈ ℝ: stiffness
dt ∈ ℝ: step size
bottom ∈ ℝ: ground height


e(i, j) = ||x_i - x_j|| where i,j ∈ ℤ vertices

computeInternalForces(i, v, xn) = tuple(vn, f+(0.0, -98.0, 0.0))  where i ∈ ℤ vertices, v_i ∈ ℝ^3, xn_i ∈ ℝ^3,
f = (sum_(j ∈ VertexOneRing(i)) (-K) (||disp|| - e(i, j)) dir
where disp = xn_i - xn_j,
dir = disp/||disp||),
vn = v_i exp(-dt damping) + dt f


applyForces(i, v, f, xn) = tuple(vn, xnn) where i ∈ ℤ vertices, v_i ∈ ℝ^3, f_i ∈ ℝ^3,xn_i ∈ ℝ^3,
a = f_i / m,
vn = v_i + a dt,
vnn = { (0, -vn_2, 0) if xn_i,2 < bottom
     vn otherwise,
xnnn = { (xn_i,1, bottom, xn_i,3) if xn_i,2 < bottom
     xn_i otherwise,
xnn = xnnn + vnn dt


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
    double K;
    double dt;
    double damping;
    double m;
    double bottom;
    double e(
        const int & i,
        const int & j)
    {
        return (x.at(i) - x.at(j)).lpNorm<2>();    
    }
    std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > computeInternalForces(
        const int & i,
        const std::vector<Eigen::Matrix<double, 3, 1>> & v,
        const std::vector<Eigen::Matrix<double, 3, 1>> & xn)
    {
        const long dim_1 = v.size();
        assert( xn.size() == dim_1 );

        Eigen::MatrixXd sum_0 = Eigen::MatrixXd::Zero(3, 1);
        for(int j : VertexOneRing(i)){
                // disp = xn_i - xn_j
            Eigen::Matrix<double, 3, 1> disp = xn.at(i) - xn.at(j);
                // dir = disp/||disp||
            Eigen::Matrix<double, 3, 1> dir = disp / double((disp).lpNorm<2>());
            sum_0 += (-K) * ((disp).lpNorm<2>() - e(i, j)) * dir;
        }
        // f = (sum_(j ∈ VertexOneRing(i)) (-K) (||disp|| - e(i, j)) dir
    // where disp = xn_i - xn_j,
    // dir = disp/||disp||)
        Eigen::Matrix<double, 3, 1> f = (sum_0);

        // vn = v_i exp(-dt damping) + dt f
        Eigen::Matrix<double, 3, 1> vn = v.at(i) * exp(-dt * damping) + dt * f;
        Eigen::Matrix<double, 3, 1> computeInternalForces_0;
        computeInternalForces_0 << 0.0, -98.0, 0.0;
        return std::tuple<Eigen::Matrix<double, 3, 1>,Eigen::Matrix<double, 3, 1> >{ vn,f + computeInternalForces_0 };    
    }
    std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > applyForces(
        const int & i,
        const std::vector<Eigen::Matrix<double, 3, 1>> & v,
        const std::vector<Eigen::Matrix<double, 3, 1>> & f,
        const std::vector<Eigen::Matrix<double, 3, 1>> & xn)
    {
        const long dim_2 = v.size();
        assert( f.size() == dim_2 );
        assert( xn.size() == dim_2 );

        // a = f_i / m
        Eigen::Matrix<double, 3, 1> a = f.at(i) / double(m);

        // vn = v_i + a dt
        Eigen::Matrix<double, 3, 1> vn = v.at(i) + a * dt;

        Eigen::Matrix<double, 3, 1> vnn;
        if(xn.at(i)[2-1] < bottom){
            Eigen::Matrix<double, 3, 1> vnn_0;
        vnn_0 << 0, -vn[2-1], 0;
            vnn = vnn_0;
        }
        else{
            vnn = vn;
        }
        // vnn = { (0, -vn_2, 0) if xn_i,2 < bottom
    //      vn otherwise

        Eigen::Matrix<double, 3, 1> xnnn;
        if(xn.at(i)[2-1] < bottom){
            Eigen::Matrix<double, 3, 1> xnnn_0;
        xnnn_0 << xn.at(i)[1-1], bottom, xn.at(i)[3-1];
            xnnn = xnnn_0;
        }
        else{
            xnnn = xn.at(i);
        }
        // xnnn = { (xn_i,1, bottom, xn_i,3) if xn_i,2 < bottom
    //      xn_i otherwise

        // xnn = xnnn + vnn dt
        Eigen::Matrix<double, 3, 1> xnn = xnnn + vnn * dt;
        return std::tuple<Eigen::Matrix<double, 3, 1>,Eigen::Matrix<double, 3, 1> >{ vn,xnn };    
    }
    struct Neighborhoods {
        std::vector<int > V;
        std::vector<int > E;
        std::vector<int > F;
        Eigen::SparseMatrix<int> dee0;
        Eigen::SparseMatrix<int> dee1;
        Eigen::SparseMatrix<int> B0;
        Eigen::SparseMatrix<int> B1;
        TriangleMesh M;
        std::vector<int > VertexOneRing(
            const int & v)
        {
            assert( std::binary_search(V.begin(), V.end(), v) );
            std::vector<int > VertexOneRingset_0({v});
            if(VertexOneRingset_0.size() > 1){
                sort(VertexOneRingset_0.begin(), VertexOneRingset_0.end());
                VertexOneRingset_0.erase(unique(VertexOneRingset_0.begin(), VertexOneRingset_0.end() ), VertexOneRingset_0.end());
            }
            std::vector<int > VertexOneRingset_1({v});
            if(VertexOneRingset_1.size() > 1){
                sort(VertexOneRingset_1.begin(), VertexOneRingset_1.end());
                VertexOneRingset_1.erase(unique(VertexOneRingset_1.begin(), VertexOneRingset_1.end() ), VertexOneRingset_1.end());
            }
            std::vector<int > difference;
            const std::vector<int >& lhs_diff = nonzeros(B0 * B0.transpose() * M.vertices_to_vector(VertexOneRingset_0));
            const std::vector<int >& rhs_diff = VertexOneRingset_1;
            difference.reserve(lhs_diff.size());
            std::set_difference(lhs_diff.begin(), lhs_diff.end(), rhs_diff.begin(), rhs_diff.end(), std::back_inserter(difference));
            return difference;    
        }
        std::vector<int > VertexOneRing(
            const std::vector<int > & v)
        {
            std::vector<int > difference_1;
            const std::vector<int >& lhs_diff_1 = nonzeros(B0 * B0.transpose() * M.vertices_to_vector(v));
            const std::vector<int >& rhs_diff_1 = v;
            difference_1.reserve(lhs_diff_1.size());
            std::set_difference(lhs_diff_1.begin(), lhs_diff_1.end(), rhs_diff_1.begin(), rhs_diff_1.end(), std::back_inserter(difference_1));
            return difference_1;    
        }
        int EdgeIndex(
            const int & i,
            const int & j)
        {
            assert( std::binary_search(V.begin(), V.end(), j) );
            std::vector<int > EdgeIndexset_0({i});
            if(EdgeIndexset_0.size() > 1){
                sort(EdgeIndexset_0.begin(), EdgeIndexset_0.end());
                EdgeIndexset_0.erase(unique(EdgeIndexset_0.begin(), EdgeIndexset_0.end() ), EdgeIndexset_0.end());
            }
            std::vector<int > EdgeIndexset_1({j});
            if(EdgeIndexset_1.size() > 1){
                sort(EdgeIndexset_1.begin(), EdgeIndexset_1.end());
                EdgeIndexset_1.erase(unique(EdgeIndexset_1.begin(), EdgeIndexset_1.end() ), EdgeIndexset_1.end());
            }
            std::vector<int > intsect;
            const std::vector<int >& lhs = nonzeros(dee0.transpose() * M.vertices_to_vector(EdgeIndexset_0));
            const std::vector<int >& rhs = nonzeros(dee0.transpose() * M.vertices_to_vector(EdgeIndexset_1));
            intsect.reserve(std::min(lhs.size(), rhs.size()));
            std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(intsect));
            std::vector<int >& stdv = intsect;
            Eigen::VectorXi vec(Eigen::Map<Eigen::VectorXi>(&stdv[0], stdv.size()));
            // evec = vec(edgeset(NonZeros(∂0ᵀ IndicatorVector(M, {i}))) ∩ vertexset(NonZeros(∂0ᵀ IndicatorVector(M, {j}))))
            Eigen::VectorXi evec = vec;
            return evec[1-1];    
        }
        std::tuple< int, int > NeighborVerticesInFace(
            const int & f,
            const int & v)
        {
            assert( std::binary_search(F.begin(), F.end(), f) );
            std::vector<int > NeighborVerticesInFaceset_0({f});
            if(NeighborVerticesInFaceset_0.size() > 1){
                sort(NeighborVerticesInFaceset_0.begin(), NeighborVerticesInFaceset_0.end());
                NeighborVerticesInFaceset_0.erase(unique(NeighborVerticesInFaceset_0.begin(), NeighborVerticesInFaceset_0.end() ), NeighborVerticesInFaceset_0.end());
            }
            // es = edgeset(NonZeros(∂1 IndicatorVector(M, {f})))
            std::vector<int > es = nonzeros(dee1 * M.faces_to_vector(NeighborVerticesInFaceset_0));
            std::vector<int > NeighborVerticesInFaceset_1;
            const std::vector<int >& range = es;
            NeighborVerticesInFaceset_1.reserve(range.size());
            for(int s : range){
                if(dee0.coeff(v, s) != 0){
                    NeighborVerticesInFaceset_1.push_back(s);
                }
            }
            if(NeighborVerticesInFaceset_1.size() > 1){
                sort(NeighborVerticesInFaceset_1.begin(), NeighborVerticesInFaceset_1.end());
                NeighborVerticesInFaceset_1.erase(unique(NeighborVerticesInFaceset_1.begin(), NeighborVerticesInFaceset_1.end() ), NeighborVerticesInFaceset_1.end());
            }
            // nes = { s for s ∈ es if ∂0_v,s != 0 }
            std::vector<int > nes = NeighborVerticesInFaceset_1;
            std::vector<int > NeighborVerticesInFaceset_2;
            const std::vector<int >& range_1 = nes;
            NeighborVerticesInFaceset_2.reserve(range_1.size());
            for(int e : range_1){
                if(dee1.coeff(e, f) * dee0.coeff(v, e) == -1){
                    NeighborVerticesInFaceset_2.push_back(e);
                }
            }
            if(NeighborVerticesInFaceset_2.size() > 1){
                sort(NeighborVerticesInFaceset_2.begin(), NeighborVerticesInFaceset_2.end());
                NeighborVerticesInFaceset_2.erase(unique(NeighborVerticesInFaceset_2.begin(), NeighborVerticesInFaceset_2.end() ), NeighborVerticesInFaceset_2.end());
            }
            // eset1 = { e for e ∈ nes if ∂1_e,f ∂0_v,e == -1}
            std::vector<int > eset1 = NeighborVerticesInFaceset_2;
            // vset1 = vertexset(NonZeros(B0 IndicatorVector(M, eset1)))
            std::vector<int > vset1 = nonzeros(B0 * M.edges_to_vector(eset1));
            std::vector<int > NeighborVerticesInFaceset_3({v});
            if(NeighborVerticesInFaceset_3.size() > 1){
                sort(NeighborVerticesInFaceset_3.begin(), NeighborVerticesInFaceset_3.end());
                NeighborVerticesInFaceset_3.erase(unique(NeighborVerticesInFaceset_3.begin(), NeighborVerticesInFaceset_3.end() ), NeighborVerticesInFaceset_3.end());
            }
            std::vector<int > difference_2;
            const std::vector<int >& lhs_diff_2 = vset1;
            const std::vector<int >& rhs_diff_2 = NeighborVerticesInFaceset_3;
            difference_2.reserve(lhs_diff_2.size());
            std::set_difference(lhs_diff_2.begin(), lhs_diff_2.end(), rhs_diff_2.begin(), rhs_diff_2.end(), std::back_inserter(difference_2));
            std::vector<int >& stdv_1 = difference_2;
            Eigen::VectorXi vec_1(Eigen::Map<Eigen::VectorXi>(&stdv_1[0], stdv_1.size()));
            // vvec1 = vec(vset1 - {v})
            Eigen::VectorXi vvec1 = vec_1;
            std::vector<int > NeighborVerticesInFaceset_4;
            const std::vector<int >& range_2 = nes;
            NeighborVerticesInFaceset_4.reserve(range_2.size());
            for(int e : range_2){
                if(dee1.coeff(e, f) * dee0.coeff(v, e) == 1){
                    NeighborVerticesInFaceset_4.push_back(e);
                }
            }
            if(NeighborVerticesInFaceset_4.size() > 1){
                sort(NeighborVerticesInFaceset_4.begin(), NeighborVerticesInFaceset_4.end());
                NeighborVerticesInFaceset_4.erase(unique(NeighborVerticesInFaceset_4.begin(), NeighborVerticesInFaceset_4.end() ), NeighborVerticesInFaceset_4.end());
            }
            // eset2 = { e for e ∈ nes if ∂1_e,f ∂0_v,e == 1 }
            std::vector<int > eset2 = NeighborVerticesInFaceset_4;
            // vset2 = vertexset(NonZeros(B0 IndicatorVector(M, eset2)))
            std::vector<int > vset2 = nonzeros(B0 * M.edges_to_vector(eset2));
            std::vector<int > NeighborVerticesInFaceset_5({v});
            if(NeighborVerticesInFaceset_5.size() > 1){
                sort(NeighborVerticesInFaceset_5.begin(), NeighborVerticesInFaceset_5.end());
                NeighborVerticesInFaceset_5.erase(unique(NeighborVerticesInFaceset_5.begin(), NeighborVerticesInFaceset_5.end() ), NeighborVerticesInFaceset_5.end());
            }
            std::vector<int > difference_3;
            const std::vector<int >& lhs_diff_3 = vset2;
            const std::vector<int >& rhs_diff_3 = NeighborVerticesInFaceset_5;
            difference_3.reserve(lhs_diff_3.size());
            std::set_difference(lhs_diff_3.begin(), lhs_diff_3.end(), rhs_diff_3.begin(), rhs_diff_3.end(), std::back_inserter(difference_3));
            std::vector<int >& stdv_2 = difference_3;
            Eigen::VectorXi vec_2(Eigen::Map<Eigen::VectorXi>(&stdv_2[0], stdv_2.size()));
            // vvec2 = vec(vset2 - {v})
            Eigen::VectorXi vvec2 = vec_2;
            return std::tuple<int,int >{ vvec1[1-1],vvec2[1-1] };    
        }
        std::tuple< int, int, int > OrientedVertices(
            const int & f)
        {
            assert( std::binary_search(F.begin(), F.end(), f) );
            // vs = Vertices(f)
            std::vector<int > vs = Vertices_0(f);
            std::vector<int >& stdv_3 = vs;
            Eigen::VectorXi vec_3(Eigen::Map<Eigen::VectorXi>(&stdv_3[0], stdv_3.size()));
            // vvec = vec(vs)
            Eigen::VectorXi vvec = vec_3;
            // i,j = NeighborVerticesInFace(f, vvec_1)
            std::tuple< int, int > tuple_3 = NeighborVerticesInFace(f, vvec[1-1]);
            int i = std::get<0>(tuple_3);
            int j = std::get<1>(tuple_3);
            return std::tuple<int,int,int >{ vvec[1-1],i,j };    
        }
        std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > Diamond(
            const int & e)
        {
            assert( std::binary_search(E.begin(), E.end(), e) );
            std::vector<int > Diamondset_0({e});
            if(Diamondset_0.size() > 1){
                sort(Diamondset_0.begin(), Diamondset_0.end());
                Diamondset_0.erase(unique(Diamondset_0.begin(), Diamondset_0.end() ), Diamondset_0.end());
            }
        std::vector<int > tetset;
            return std::tuple<std::vector<int >,std::vector<int >,std::vector<int >,std::vector<int > >{ Vertices_2(e),Diamondset_0,Faces_1(e),tetset };    
        }
        std::tuple< int, int > OppositeVertices(
            const int & e)
        {
            assert( std::binary_search(E.begin(), E.end(), e) );
            std::vector<int > difference_4;
            const std::vector<int >& lhs_diff_4 = Vertices_1(Faces_1(e));
            const std::vector<int >& rhs_diff_4 = Vertices_2(e);
            difference_4.reserve(lhs_diff_4.size());
            std::set_difference(lhs_diff_4.begin(), lhs_diff_4.end(), rhs_diff_4.begin(), rhs_diff_4.end(), std::back_inserter(difference_4));
            std::vector<int >& stdv_4 = difference_4;
            Eigen::VectorXi vec_4(Eigen::Map<Eigen::VectorXi>(&stdv_4[0], stdv_4.size()));
            // evec = vec(Vertices(Faces(e)) \ Vertices(e))
            Eigen::VectorXi evec = vec_4;
            return std::tuple<int,int >{ evec[1-1],evec[2-1] };    
        }
        int FaceIndex(
            const int & i,
            const int & j,
            const int & k)
        {
            assert( std::binary_search(V.begin(), V.end(), k) );
            // ufv = (B0 B1)ᵀ
            Eigen::SparseMatrix<int> ufv = (B0 * B1).transpose();
            std::vector<int > FaceIndexset_0({i});
            if(FaceIndexset_0.size() > 1){
                sort(FaceIndexset_0.begin(), FaceIndexset_0.end());
                FaceIndexset_0.erase(unique(FaceIndexset_0.begin(), FaceIndexset_0.end() ), FaceIndexset_0.end());
            }
            // iface = faceset(NonZeros(ufv  IndicatorVector(M, {i})))
            std::vector<int > iface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_0));
            std::vector<int > FaceIndexset_1({j});
            if(FaceIndexset_1.size() > 1){
                sort(FaceIndexset_1.begin(), FaceIndexset_1.end());
                FaceIndexset_1.erase(unique(FaceIndexset_1.begin(), FaceIndexset_1.end() ), FaceIndexset_1.end());
            }
            // jface = faceset(NonZeros(ufv  IndicatorVector(M, {j})))
            std::vector<int > jface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_1));
            std::vector<int > FaceIndexset_2({k});
            if(FaceIndexset_2.size() > 1){
                sort(FaceIndexset_2.begin(), FaceIndexset_2.end());
                FaceIndexset_2.erase(unique(FaceIndexset_2.begin(), FaceIndexset_2.end() ), FaceIndexset_2.end());
            }
            // kface = faceset(NonZeros(ufv IndicatorVector(M, {k})))
            std::vector<int > kface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_2));
            std::vector<int > intsect_1;
            const std::vector<int >& lhs_1 = jface;
            const std::vector<int >& rhs_1 = kface;
            intsect_1.reserve(std::min(lhs_1.size(), rhs_1.size()));
            std::set_intersection(lhs_1.begin(), lhs_1.end(), rhs_1.begin(), rhs_1.end(), std::back_inserter(intsect_1));
            std::vector<int > intsect_2;
            const std::vector<int >& lhs_2 = iface;
            const std::vector<int >& rhs_2 = intsect_1;
            intsect_2.reserve(std::min(lhs_2.size(), rhs_2.size()));
            std::set_intersection(lhs_2.begin(), lhs_2.end(), rhs_2.begin(), rhs_2.end(), std::back_inserter(intsect_2));
            std::vector<int >& stdv_5 = intsect_2;
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
            assert( std::binary_search(V.begin(), V.end(), k) );
            // f = FaceIndex(i, j, k)
            int f = FaceIndex(i, j, k);
            return NeighborVerticesInFace(f, i);    
        }
        struct FundamentalMeshAccessors {
            std::vector<int > V;
            std::vector<int > E;
            std::vector<int > F;
            Eigen::SparseMatrix<int> dee0;
            Eigen::SparseMatrix<int> dee1;
            Eigen::SparseMatrix<int> B0;
            Eigen::SparseMatrix<int> B1;
            TriangleMesh M;
            std::vector<int > Vertices(
                const std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > & S)
            {
                return std::get<1-1>(S);    
            }
            std::vector<int > Edges(
                const std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > & S)
            {
                return std::get<2-1>(S);    
            }
            std::vector<int > Faces(
                const std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > & S)
            {
                return std::get<3-1>(S);    
            }
            std::vector<int > Vertices_0(
                const int & f)
            {
                assert( std::binary_search(F.begin(), F.end(), f) );
                std::vector<int > Vertices_0set_0({f});
                if(Vertices_0set_0.size() > 1){
                    sort(Vertices_0set_0.begin(), Vertices_0set_0.end());
                    Vertices_0set_0.erase(unique(Vertices_0set_0.begin(), Vertices_0set_0.end() ), Vertices_0set_0.end());
                }
                return nonzeros(B0 * B1 * M.faces_to_vector(Vertices_0set_0));    
            }
            std::vector<int > Vertices_1(
                const std::vector<int > & G)
            {
                return nonzeros(B0 * B1 * M.faces_to_vector(G));    
            }
            std::vector<int > Vertices_2(
                const int & e)
            {
                assert( std::binary_search(E.begin(), E.end(), e) );
                std::vector<int > Vertices_2set_0({e});
                if(Vertices_2set_0.size() > 1){
                    sort(Vertices_2set_0.begin(), Vertices_2set_0.end());
                    Vertices_2set_0.erase(unique(Vertices_2set_0.begin(), Vertices_2set_0.end() ), Vertices_2set_0.end());
                }
                return nonzeros(B0 * M.edges_to_vector(Vertices_2set_0));    
            }
            std::vector<int > Vertices_3(
                const std::vector<int > & H)
            {
                return nonzeros(B0 * M.edges_to_vector(H));    
            }
            std::vector<int > Edges_0(
                const int & v)
            {
                assert( std::binary_search(V.begin(), V.end(), v) );
                std::vector<int > Edges_0set_0({v});
                if(Edges_0set_0.size() > 1){
                    sort(Edges_0set_0.begin(), Edges_0set_0.end());
                    Edges_0set_0.erase(unique(Edges_0set_0.begin(), Edges_0set_0.end() ), Edges_0set_0.end());
                }
                return nonzeros(B0.transpose() * M.vertices_to_vector(Edges_0set_0));    
            }
            std::vector<int > Edges_1(
                const int & f)
            {
                assert( std::binary_search(F.begin(), F.end(), f) );
                std::vector<int > Edges_1set_0({f});
                if(Edges_1set_0.size() > 1){
                    sort(Edges_1set_0.begin(), Edges_1set_0.end());
                    Edges_1set_0.erase(unique(Edges_1set_0.begin(), Edges_1set_0.end() ), Edges_1set_0.end());
                }
                return nonzeros(B1 * M.faces_to_vector(Edges_1set_0));    
            }
            std::vector<int > Faces_0(
                const int & v)
            {
                assert( std::binary_search(V.begin(), V.end(), v) );
                std::vector<int > Faces_0set_0({v});
                if(Faces_0set_0.size() > 1){
                    sort(Faces_0set_0.begin(), Faces_0set_0.end());
                    Faces_0set_0.erase(unique(Faces_0set_0.begin(), Faces_0set_0.end() ), Faces_0set_0.end());
                }
                return nonzeros((B0 * B1).transpose() * M.vertices_to_vector(Faces_0set_0));    
            }
            std::vector<int > Faces_1(
                const int & e)
            {
                assert( std::binary_search(E.begin(), E.end(), e) );
                std::vector<int > Faces_1set_0({e});
                if(Faces_1set_0.size() > 1){
                    sort(Faces_1set_0.begin(), Faces_1set_0.end());
                    Faces_1set_0.erase(unique(Faces_1set_0.begin(), Faces_1set_0.end() ), Faces_1set_0.end());
                }
                return nonzeros(B1.transpose() * M.edges_to_vector(Faces_1set_0));    
            }
            FundamentalMeshAccessors(const TriangleMesh & M)
            {
                // V, E, F = MeshSets( M )
                std::tuple< std::vector<int >, std::vector<int >, std::vector<int > > tuple = M.MeshSets();
                V = std::get<0>(tuple);
                E = std::get<1>(tuple);
                F = std::get<2>(tuple);
                int dimv_0 = M.n_vertices();
                int dime_0 = M.n_edges();
                int dimf_0 = M.n_faces();
                this->M = M;
                // dee0, dee1 = BoundaryMatrices(M)
                std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_1 = M.BoundaryMatrices();
                dee0 = std::get<0>(tuple_1);
                dee1 = std::get<1>(tuple_1);
                // B0, B1 = UnsignedBoundaryMatrices(M)
                std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_2 = M.UnsignedBoundaryMatrices();
                B0 = std::get<0>(tuple_2);
                B1 = std::get<1>(tuple_2);
            }
        };
        FundamentalMeshAccessors _FundamentalMeshAccessors;
        std::vector<int > Vertices(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
            return _FundamentalMeshAccessors.Vertices(p0);
        };
        std::vector<int > Vertices_0(int p0){
            return _FundamentalMeshAccessors.Vertices_0(p0);
        };
        std::vector<int > Vertices_1(std::vector<int > p0){
            return _FundamentalMeshAccessors.Vertices_1(p0);
        };
        std::vector<int > Vertices_2(int p0){
            return _FundamentalMeshAccessors.Vertices_2(p0);
        };
        std::vector<int > Vertices_3(std::vector<int > p0){
            return _FundamentalMeshAccessors.Vertices_3(p0);
        };
        std::vector<int > Edges(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
            return _FundamentalMeshAccessors.Edges(p0);
        };
        std::vector<int > Edges_0(int p0){
            return _FundamentalMeshAccessors.Edges_0(p0);
        };
        std::vector<int > Edges_1(int p0){
            return _FundamentalMeshAccessors.Edges_1(p0);
        };
        std::vector<int > Faces(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
            return _FundamentalMeshAccessors.Faces(p0);
        };
        std::vector<int > Faces_0(int p0){
            return _FundamentalMeshAccessors.Faces_0(p0);
        };
        std::vector<int > Faces_1(int p0){
            return _FundamentalMeshAccessors.Faces_1(p0);
        };
        Neighborhoods(const TriangleMesh & M)
        :
        _FundamentalMeshAccessors(M)
        {
            // V, E, F = MeshSets( M )
            std::tuple< std::vector<int >, std::vector<int >, std::vector<int > > tuple = M.MeshSets();
            V = std::get<0>(tuple);
            E = std::get<1>(tuple);
            F = std::get<2>(tuple);
            int dimv_0 = M.n_vertices();
            int dime_0 = M.n_edges();
            int dimf_0 = M.n_faces();
            this->M = M;
            // ∂0, ∂1 = BoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_1 = M.BoundaryMatrices();
            dee0 = std::get<0>(tuple_1);
            dee1 = std::get<1>(tuple_1);
            // B0, B1 = UnsignedBoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_2 = M.UnsignedBoundaryMatrices();
            B0 = std::get<0>(tuple_2);
            B1 = std::get<1>(tuple_2);
        }
    };
    Neighborhoods _Neighborhoods;
    std::vector<int > VertexOneRing(int p0){
        return _Neighborhoods.VertexOneRing(p0);
    };
    std::vector<int > VertexOneRing(std::vector<int > p0){
        return _Neighborhoods.VertexOneRing(p0);
    };
    iheartla(
        const TriangleMesh & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x,
        const double & m,
        const double & damping,
        const double & K,
        const double & dt,
        const double & bottom)
    :
    _Neighborhoods(M)
    {
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        int dimf_0 = M.n_faces();
        const long dim_0 = x.size();
        this->x = x;
        this->K = K;
        this->dt = dt;
        this->damping = damping;
        this->m = m;
        this->bottom = bottom;
    
    }
};













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
    iheartla ihla(triangle_mesh, OriginalPosition, mass, damping, stiffness, dt, bottom_z);
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
    igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_disk.obj", meshV, meshF);
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
