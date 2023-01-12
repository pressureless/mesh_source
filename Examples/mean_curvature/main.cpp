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
#include "MeshHelper.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
/*
FaceNeighbors, EdgeIndex, VertexOneRing, OppositeVertices, OrientedVertices, NeighborVerticesInFace from Neighborhoods(M)
M ∈ mesh

t ∈ ℝ: step length

clamp(v) = { -bound if v < -bound
             bound if v > bound
         v otherwise where v ∈ ℝ,
bound = 19.1

area(f, p, x) = { 0 if A=0
        ½A if dotp < 0
                  ¼A if dotq <0 or dotr <0
                  ⅛(cotq ||pr||² + cotr ||pq||² ) otherwise where f ∈ ℤ faces, x_i ∈ ℝ^3, p ∈ ℤ vertices,
q,r = NeighborVerticesInFace(f, p),
pq = x_q - x_p,
qr = x_r - x_q,
pr = x_r - x_p,
A = ½||pq×pr||,
dotp = pq ⋅ pr,
dotq = (x_q-x_r) ⋅ pq,
dotr = qr⋅ pr,
cotq = clamp(dotq/(2A)),
cotr = clamp(dotr/(2A))

 
cot(k, j, i, x) = { clamp(cos/sin) if sin≠0
                    0 otherwise where i,j,k ∈ ℤ vertices, x_i ∈ ℝ^3 ,
oj, oi = OrientedVertices(k, j, i),
cos = (x_oj - x_k)⋅(x_oi-x_k),
sin = ||(x_oj - x_k)×(x_oi-x_k)||


Ax(i, x) = x_i - t w K  where i ∈ ℤ vertices, x_i ∈ ℝ^3,
A = (sum_(f ∈ FaceNeighbors(i)) area(f, i, x)),
w = { 1/(2A) if A≠0
      0      otherwise,
K = (sum_(j ∈ VertexOneRing(i)) max(`cot(α)` + `cot(β)`,0)(x_j - x_i) 
where k, l = OppositeVertices(EdgeIndex(i,j)),
`cot(α)` = cot(k, j, i, x),
`cot(β)` = cot(l, i, j, x) )

*/
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include <algorithm>

struct iheartla {

    double t;
    double clamp(
        const double & v)
    {
        double clamp_ret;
        // bound = 19.1
        double bound = 19.1;
        if(v < -bound){
            clamp_ret = -bound;
        }
        else if(v > bound){
            clamp_ret = bound;
        }
        else{
            clamp_ret = v;
        }
        return clamp_ret;    
    }
    double area(
        const int & f,
        const int & p,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    {
        const long dim_0 = x.size();
        double area_ret;
        // q,r = NeighborVerticesInFace(f, p)
        std::tuple< int, int > tuple = NeighborVerticesInFace(f, p);
        int q = std::get<0>(tuple);
        int r = std::get<1>(tuple);

        // pq = x_q - x_p
        Eigen::Matrix<double, 3, 1> pq = x.at(q) - x.at(p);

        // qr = x_r - x_q
        Eigen::Matrix<double, 3, 1> qr = x.at(r) - x.at(q);

        // pr = x_r - x_p
        Eigen::Matrix<double, 3, 1> pr = x.at(r) - x.at(p);

        // A = ½||pq×pr||
        double A = (1/double(2)) * ((pq).cross(pr)).lpNorm<2>();

        // dotp = pq ⋅ pr
        double dotp = (pq).dot(pr);

        // dotq = (x_q-x_r) ⋅ pq
        double dotq = ((x.at(q) - x.at(r))).dot(pq);

        // dotr = qr⋅ pr
        double dotr = (qr).dot(pr);

        // cotq = clamp(dotq/(2A))
        double cotq = clamp(dotq / double((2 * A)));

        // cotr = clamp(dotr/(2A))
        double cotr = clamp(dotr / double((2 * A)));
        if(A == 0){
            area_ret = 0;
        }
        else if(dotp < 0){
            area_ret = (1/double(2)) * A;
        }
        else if((dotq < 0) || (dotr < 0)){
            area_ret = (1/double(4)) * A;
        }
        else{
            area_ret = (1/double(8)) * (cotq * pow((pr).lpNorm<2>(), 2) + cotr * pow((pq).lpNorm<2>(), 2));
        }
        return area_ret;    
    }
    double cot(
        const int & k,
        const int & j,
        const int & i,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    {
        const long dim_1 = x.size();
        double cot_ret;
        // oj, oi = OrientedVertices(k, j, i)
        std::tuple< int, int > tuple_1 = OrientedVertices(k, j, i);
        int oj = std::get<0>(tuple_1);
        int oi = std::get<1>(tuple_1);

        // cos = (x_oj - x_k)⋅(x_oi-x_k)
        double cos = ((x.at(oj) - x.at(k))).dot((x.at(oi) - x.at(k)));

        // sin = ||(x_oj - x_k)×(x_oi-x_k)||
        double sin = (((x.at(oj) - x.at(k))).cross((x.at(oi) - x.at(k)))).lpNorm<2>();
        if(sin != 0){
            cot_ret = clamp(cos / double(sin));
        }
        else{
            cot_ret = 0;
        }
        return cot_ret;    
    }
    Eigen::Matrix<double, 3, 1> Ax(
        const int & i,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    {
        const long dim_2 = x.size();
        double sum_0 = 0;
        for(int f : FaceNeighbors(i)){
            sum_0 += area(f, i, x);
        }
        // A = (sum_(f ∈ FaceNeighbors(i)) area(f, i, x))
        double A = (sum_0);

        double w;
        if(A != 0){
            w = 1 / double((2 * A));
        }
        else{
            w = 0;
        }
        // w = { 1/(2A) if A≠0
    //       0      otherwise

        Eigen::MatrixXd sum_1 = Eigen::MatrixXd::Zero(3, 1);
        for(int j : VertexOneRing(i)){
                // k, l = OppositeVertices(EdgeIndex(i,j))
            std::tuple< int, int > tuple_2 = OppositeVertices(EdgeIndex(i, j));
            int k = std::get<0>(tuple_2);
            int l = std::get<1>(tuple_2);
                // `cot(α)` = cot(k, j, i, x)
            double cotα = cot(k, j, i, x);
                // `cot(β)` = cot(l, i, j, x)
            double cotβ = cot(l, i, j, x);
            sum_1 += std::max({double(cotα + cotβ), double(0)}) * (x.at(j) - x.at(i));
        }
        // K = (sum_(j ∈ VertexOneRing(i)) max(`cot(α)` + `cot(β)`,0)(x_j - x_i) 
    // where k, l = OppositeVertices(EdgeIndex(i,j)),
    // `cot(α)` = cot(k, j, i, x),
    // `cot(β)` = cot(l, i, j, x) )
        Eigen::Matrix<double, 3, 1> K = (sum_1);
        return x.at(i) - t * w * K;    
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
    std::vector<int > FaceNeighbors(int p0){
        return _Neighborhoods.FaceNeighbors(p0);
    };
    std::vector<int > FaceNeighbors_0(int p0){
        return _Neighborhoods.FaceNeighbors_0(p0);
    };
    int EdgeIndex(int p0,int p1){
        return _Neighborhoods.EdgeIndex(p0,p1);
    };
    std::vector<int > VertexOneRing(int p0){
        return _Neighborhoods.VertexOneRing(p0);
    };
    std::vector<int > VertexOneRing(std::vector<int > p0){
        return _Neighborhoods.VertexOneRing(p0);
    };
    std::tuple< int, int > OppositeVertices(int p0){
        return _Neighborhoods.OppositeVertices(p0);
    };
    std::tuple< int, int, int > OrientedVertices(int p0){
        return _Neighborhoods.OrientedVertices(p0);
    };
    std::tuple< int, int > OrientedVertices(int p0,int p1,int p2){
        return _Neighborhoods.OrientedVertices(p0,p1,p2);
    };
    std::tuple< int, int > NeighborVerticesInFace(int p0,int p1){
        return _Neighborhoods.NeighborVerticesInFace(p0,p1);
    };
    iheartla(
        const TriangleMesh & M,
        const double & t)
    :
    _Neighborhoods(M)
    {
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        int dimf_0 = M.n_faces();
        this->t = t;
    
    }
};












Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
TriangleMesh triangle_mesh;

double cg_tolerance = 1e-3;
int MAX_ITERATION = 10;
double step = 5e-3;

std::vector<Eigen::Matrix<double, 3, 1>> OriginalPosition;
std::vector<Eigen::Matrix<double, 3, 1>> Position;

void axpy3(const std::vector<Eigen::Matrix<double, 3, 1>>& X,
           const double                            alpha,
           const double                            beta,
           std::vector<Eigen::Matrix<double, 3, 1>>&       Y)
{
    // Y = beta*Y + alpha*X 
    int size = static_cast<int>(X.size()); 
    for (int i = 0; i < size; ++i) {
        Y[i] *= beta;
        Y[i] += alpha * X[i];
    }
}

double dot3(const std::vector<Eigen::Matrix<double, 3, 1>>& A,
       const std::vector<Eigen::Matrix<double, 3, 1>>& B)
{

    double  ret = 0;
    int size = static_cast<int>(A.size());
    for (int i = 0; i < size; ++i) {
        ret += A[i].dot(B[i]);
    }
    return ret;
}


void conjugate_gradients_algorithm(std::vector<Eigen::Matrix<double, 3, 1>>& X,
        std::vector<Eigen::Matrix<double, 3, 1>>& B,
        std::vector<Eigen::Matrix<double, 3, 1>>& R,
        std::vector<Eigen::Matrix<double, 3, 1>>& P,
        std::vector<Eigen::Matrix<double, 3, 1>>& S,
        double& start_residual,
        double&  stop_residual,
        iheartla& ihla)
{
    // Conjugate Gradients
    // Page 50 in "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"
    // CG solver. Solve for the three coordinates simultaneously

    // s = Ax 
    for (int i = 0; i < meshV.rows(); ++i)
    {
        S[i] = ihla.Ax(i, X);
    } 
 
    // r = b - s = b - Ax
    // p = r 
    for (int i = 0; i < int(meshV.rows()); ++i) {
        R[i] = B[i] - S[i];
        R[i] = B[i] - S[i];
        R[i] = B[i] - S[i];

        P[i] = R[i];
        P[i] = R[i];
        P[i] = R[i];
    }
 
    // delta_new = <r,r>
    double delta_new = dot3(R, R);

    // delta_0 = delta_new
    double delta_0 = delta_new;
        std::cout<<"delta_new delta_new: "<<delta_new<<std::endl;

    start_residual = delta_0;
    uint32_t iter  = 0;
    while (iter < MAX_ITERATION) { 
        // s = Ap 
        for (int i = 0; i < meshV.rows(); ++i)
        {
            S[i] = ihla.Ax(i, P);
        }  

        // alpha = delta_new / <s,p>
        double alpha = dot3(S, P);
        std::cout<<"alpha before: "<<alpha<<std::endl;
        alpha = delta_new / alpha;

        std::cout<<"alpha: "<<alpha<<std::endl;
        // x =  x + alpha*p
        axpy3(P, alpha, 1.0, X);

        // r = r - alpha*s
        axpy3(S, -alpha, 1.0, R);

        // delta_old = delta_new
        double delta_old(delta_new);

        // delta_new = <r,r>
        delta_new = dot3(R, R);

        // beta = delta_new/delta_old
        double beta(delta_new / delta_old);

        // exit if error is getting too low across three coordinates
        std::cout<<"delta_new:"<<delta_new<<std::endl;
        std::cout<<"cg_tolerance * cg_tolerance * delta_0:"<<cg_tolerance * cg_tolerance * delta_0<<std::endl;
            
        if (delta_new < cg_tolerance * cg_tolerance * delta_0) {
            break;
        }

        // p = beta*p + r
        axpy3(R, 1.0, beta, P);

        ++iter;
        std::cout<<"iter:"<<iter<<std::endl;
    }  
    stop_residual = delta_new;
}

void update(){
    iheartla ihla(triangle_mesh, step); 
    std::cout<<"After"<<std::endl;
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout<<"i: "<<i<<", "<<P[i]<<std::endl;
    // } 
    double start_residual, stop_residual;
    std::vector<Eigen::Matrix<double, 3, 1>> X(meshV.rows()), B(meshV.rows()), R(meshV.rows()), P(meshV.rows()), S(meshV.rows());

    for (int i = 0; i < meshV.rows(); ++i)
    {
        X[i] = Position[i];
        B[i] = Position[i];
    } 
    conjugate_gradients_algorithm(X, B, R, P, S, start_residual, stop_residual, ihla);
    std::cout<<"start_residual:"<<start_residual<<", stop_residual:"<<stop_residual<<std::endl;
    
    Position = X;
    double min_diff = 1000;
    double max_diff = 0;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        double norm = (X[i]-OriginalPosition[i]).norm();
        if (norm < min_diff)
        {
            min_diff = norm;
        }
        if(norm > max_diff){
            max_diff = norm;
        }
        // std::cout<<"i: "<<i<<", ("<<X[i][0]-OriginalPosition[i][0]<<", "<<X[i][1]-OriginalPosition[i][1]<<", "<<X[i][2]-OriginalPosition[i][2]<<")"<<std::endl;
    } 
    std::cout<<"After updating, min_offset: "<<min_diff<<", max_offset: "<<max_diff<<std::endl;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(Position);
}


void myCallback()
{ 
    if (ImGui::Button("One step")){ 
        update();
    } 
    if (ImGui::Button("Ten steps")){
        for (int i = 0; i < 10; ++i)
        {
            update();
        }
    } 
    if (ImGui::Button("Fifty steps")){
        for (int i = 0; i < 50; ++i)
        {
            update();
        }
    } 
}

 

int main(int argc, const char * argv[]) {
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere3.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/libigl-polyscope-project/input/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/ddg-exercises/input/sphere.obj", meshV, meshF);
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/bumpy.off", meshV, meshF); // 69KB 5mins
    
    igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/cow.off", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    triangle_mesh.initialize(meshF);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::state::userCallback = myCallback;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Position.push_back(meshV.row(i).transpose());
        OriginalPosition.push_back(meshV.row(i).transpose());
    } 
    // for (int i = 0; i < meshV.rows(); ++i)
    // {
    //     std::cout<<"i: "<<i<<", ("<<Position[i][0]<<", "<<Position[i][1]<<", "<<Position[i][2]<<")"<<std::endl;
    // } 
    polyscope::show();
    return 0;
}
