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
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
/*
VertexOneRing, FaceNeighbors from Neighborhoods(M)
M ∈ mesh
x_i ∈ ℝ^3  


Min(a, b) = { a  if a < b
              b  otherwise where a,b∈ ℝ


UpdateStep(v0, v1, v2, d) = { p if s_1,1 < 0 and s_2,1 < 0 
            Min(d_(v1)+||x1||, d_(v2)+||x2||) otherwise where v0,v1,v2 ∈ ℤ vertices, d_i ∈ ℝ,
x1 = x_(v1) - x_(v0),
x2 = x_(v2) - x_(v0),
X = [x1 x2],
t = [d_(v1) d_(v2)]ᵀ,
Q = (XᵀX)⁻¹,
`1` = [1 ; 1],
p = (`1`ᵀQt + sqrt((`1`ᵀQt)² - `1`ᵀQ`1` ⋅ (tᵀQt - 1)))/ (`1`ᵀQ`1`),
n = XQ(t- p ⋅`1`),
s = QXᵀn

GetNextLevel(U) = v - s where U_i  ∈  {ℤ} vertices,
s = ∪_i U_i,
v = VertexOneRing(s)

GetRangeLevel(U, a, b) = ∪_(i=a)^b U_i where U_j  ∈  {ℤ} vertices, a,b ∈ ℤ index


GetLevelSequence(U) = { sequence(U, n) if |n| ≠ 0
                        U otherwise where U_i  ∈  {ℤ} vertices,
n = GetNextLevel(U)
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
    double Min(
        const double & a,
        const double & b)
    {
        double Min_ret;
        if(a < b){
            Min_ret = a;
        }
        else{
            Min_ret = b;
        }
        return Min_ret;    
    }
    double UpdateStep(
        const int & v0,
        const int & v1,
        const int & v2,
        const std::vector<double> & d)
    {
        const long dim_1 = d.size();
        double UpdateStep_ret;
        // x1 = x_(v1) - x_(v0)
        Eigen::Matrix<double, 3, 1> x1 = x.at(v1) - x.at(v0);

        // x2 = x_(v2) - x_(v0)
        Eigen::Matrix<double, 3, 1> x2 = x.at(v2) - x.at(v0);

        Eigen::Matrix<double, 3, 2> X_0;
        X_0 << x1, x2;
        // X = [x1 x2]
        Eigen::Matrix<double, 3, 2> X = X_0;

        Eigen::Matrix<double, 1, 2> t_0;
        t_0 << d.at(v1), d.at(v2);
        // t = [d_(v1) d_(v2)]ᵀ
        Eigen::Matrix<double, 2, 1> t = t_0.transpose();

        // Q = (XᵀX)⁻¹
        Eigen::Matrix<double, 2, 2> Q = (X.transpose() * X).inverse();

        Eigen::Matrix<int, 2, 1> num1_0;
        num1_0 << 1,
        1;
        // `1` = [1 ; 1]
        Eigen::Matrix<int, 2, 1> num1 = num1_0;

        // p = (`1`ᵀQt + sqrt((`1`ᵀQt)² - `1`ᵀQ`1` ⋅ (tᵀQt - 1)))/ (`1`ᵀQ`1`)
        double p = ((double)((num1.transpose()).cast<double>() * Q * t) + sqrt(pow(((double)((num1.transpose()).cast<double>() * Q * t)), 2) - (double)((num1.transpose()).cast<double>() * Q * (num1).cast<double>()) * ((double)(t.transpose() * Q * t) - 1))) / double(((double)((num1.transpose()).cast<double>() * Q * (num1).cast<double>())));

        // n = XQ(t- p ⋅`1`)
        Eigen::Matrix<double, 3, 1> n = X * Q * (t - (p * (num1).cast<double>()).cast<double>());

        // s = QXᵀn
        Eigen::Matrix<double, 2, 1> s = Q * X.transpose() * n;
        if((s(1-1, 1-1) < 0) && (s(2-1, 1-1) < 0)){
            UpdateStep_ret = p;
        }
        else{
            UpdateStep_ret = Min(d.at(v1) + (x1).lpNorm<2>(), d.at(v2) + (x2).lpNorm<2>());
        }
        return UpdateStep_ret;    
    }
    std::vector<int > GetNextLevel(
        const std::vector<std::vector<int >> & U)
    {
        const long dim_2 = U.size();
        std::cout<<"current GetNextLevel: "<<std::endl;
        std::cout<<"U size: "<<U.size()<<std::endl;
        for (int i = 0; i < U.size(); ++i)
        {
           std::cout<<"current level: "<<i<<", size:"<<U[i].size()<<std::endl;
           for (int j = 0; j < U[i].size(); ++j)
           {
               std::cout<<U[i][j]<<", ";
           }
            std::cout<<std::endl;
        }
        std::vector<int > union_0;
        for(int i=1; i<=U.size(); i++){
            std::set_union(union_0.begin(), union_0.end(), U.at(i-1).begin(), U.at(i-1).end(), std::back_inserter(union_0));
        }
        // s = ∪_i U_i
        std::vector<int > s = union_0;

        // v = VertexOneRing(s)
        std::vector<int > v = VertexOneRing(s);
        std::vector<int > difference;
        std::vector<int > lhs_diff = v;
        std::vector<int > rhs_diff = s;
        difference.reserve(lhs_diff.size());
        std::set_difference(lhs_diff.begin(), lhs_diff.end(), rhs_diff.begin(), rhs_diff.end(), std::back_inserter(difference));
        return difference;    
    }
    std::vector<int > GetRangeLevel(
        const std::vector<std::vector<int >> & U,
        const int & a,
        const int & b)
    {
        const long dim_3 = U.size();
        std::vector<int > union_1;
        for(int i=a; i<=b; i++){
            std::set_union(union_1.begin(), union_1.end(), U.at(i).begin(), U.at(i).end(), std::back_inserter(union_1));
        }
        return union_1;    
    }
    std::vector<std::vector<int >> GetLevelSequence(
        const std::vector<std::vector<int >> & U)
    {
        const long dim_4 = U.size();
        std::vector<std::vector<int >> GetLevelSequence_ret;
        // n = GetNextLevel(U)
        std::vector<int > n = GetNextLevel(U);
        if((n).size() != 0){
            std::vector<std::vector<int >> seq = U;
            seq.reserve(seq.size()+1);
            seq.push_back(n);
            GetLevelSequence_ret = seq;
        }
        else{
            GetLevelSequence_ret = U;
        }
        return GetLevelSequence_ret;    
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
        std::vector<int > FaceNeighbors(
            const int & v)
        {
            assert( std::binary_search(V.begin(), V.end(), v) );
            std::vector<int > FaceNeighborsset_0({v});
            if(FaceNeighborsset_0.size() > 1){
                sort(FaceNeighborsset_0.begin(), FaceNeighborsset_0.end());
                FaceNeighborsset_0.erase(unique(FaceNeighborsset_0.begin(), FaceNeighborsset_0.end() ), FaceNeighborsset_0.end());
            }
            return nonzeros((B0 * B1).transpose() * M.vertices_to_vector(FaceNeighborsset_0));    
        }
        std::vector<int > FaceNeighbors_0(
            const int & e)
        {
            assert( std::binary_search(E.begin(), E.end(), e) );
            std::vector<int > FaceNeighbors_0set_0({e});
            if(FaceNeighbors_0set_0.size() > 1){
                sort(FaceNeighbors_0set_0.begin(), FaceNeighbors_0set_0.end());
                FaceNeighbors_0set_0.erase(unique(FaceNeighbors_0set_0.begin(), FaceNeighbors_0set_0.end() ), FaceNeighbors_0set_0.end());
            }
            return nonzeros(B1.transpose() * M.edges_to_vector(FaceNeighbors_0set_0));    
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
            return std::tuple<std::vector<int >,std::vector<int >,std::vector<int >,std::vector<int > >{ Vertices_2(e),Diamondset_0,FaceNeighbors_0(e),tetset };    
        }
        std::tuple< int, int > OppositeVertices(
            const int & e)
        {
            assert( std::binary_search(E.begin(), E.end(), e) );
            std::vector<int > difference_4;
            const std::vector<int >& lhs_diff_4 = Vertices_1(FaceNeighbors_0(e));
            const std::vector<int >& rhs_diff_4 = Vertices_2(e);
            difference_4.reserve(lhs_diff_4.size());
            std::set_difference(lhs_diff_4.begin(), lhs_diff_4.end(), rhs_diff_4.begin(), rhs_diff_4.end(), std::back_inserter(difference_4));
            std::vector<int >& stdv_4 = difference_4;
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
    std::vector<int > FaceNeighbors(int p0){
        return _Neighborhoods.FaceNeighbors(p0);
    };
    std::vector<int > FaceNeighbors_0(int p0){
        return _Neighborhoods.FaceNeighbors_0(p0);
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
    igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
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
    iheartla ihla(triangle_mesh, P);
    std::vector<double > distance;
    std::vector<Eigen::Matrix<double, 3, 1>> N;
    for (int i = 0; i < meshV.rows(); ++i)
    { 
        distance.push_back(10000); 
    } 
 
    int cur = 316;
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
            std::vector<int> f_set = ihla.FaceNeighbors(v);
            for (int f: f_set)
            {
                std::tuple< int, int > v_tuple = ihla._Neighborhoods.NeighborVerticesInFace(f, v);
                int v1 = std::get<0>(v_tuple);
                int v2 = std::get<1>(v_tuple);
                // std::cout<<"f: "<<f<<", v:("<<v<<", "<<v1<<", "<<v2<<")"<<std::endl;
                double updated = ihla.UpdateStep(v, v1, v2, distance);
                // std::cout<<"updated: "<<updated<<std::endl;
                new_distance[v] = ihla.Min(new_distance[v], updated);
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
