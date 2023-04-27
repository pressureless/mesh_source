/*
vec, inversevec, diag, svd from linearalgebra
ElementSets from MeshConnectivity
NeighborVerticesInFace, Faces, Vertices, VertexOneRing, OrientedVertices from TetrahderonNeighborhoods(M)
M : TetrahedralMesh
x̄_i ∈ ℝ^3 : rest pos
x_i ∈ ℝ^3 : current pos
bx_j ∈ ℤ index: boundary indices
bp_j ∈ ℝ^3 : boundary positions
w ∈ ℝ : penalty
ε ∈ ℝ : eps

V, E, F, C = ElementSets(M)

vol_i,j,k,l = ⅙ |[x̄_j-x̄_i x̄_k-x̄_i x̄_l-x̄_i]| where i,j,k,l ∈ V

psd(x) = u diag(ps)  vᵀ where x ∈ ℝ^(p×p),
u, sigma, v = svd(x),
ps_i = { sigma_i if sigma_i > 0
     0 otherwise

S(s, x) = vol_abcd (‖J‖² + ‖J⁻¹‖²) where s ∈ C, x_i ∈ ℝ^3,
a, b, c, d = Vertices(s),
J = [x_b-x_a x_c-x_a x_d-x_a][x̄_b-x̄_a x̄_c-x̄_a x̄_d-x̄_a]⁻¹

E2 = w sum_j ‖bp_j - x_(bx_j)‖

e = sum_(i ∈ C) S(i, x) + E2

G = ∂e/∂x
H = sum_(i ∈ C) psd(∂²S(i, x)/∂x²) + ∂²E2/∂x²

d = H⁻¹ (-G)

y = { inversevec(vec(x) + 0.1 d, x) if √(-d⋅G) > ε
      x otherwise

*/
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include <algorithm>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include "type_helper.h"
#include "Tetrahedron.h"

template<class DT = autodiff::var, class MatrixD = Eigen::Matrix<autodiff::var, Eigen::Dynamic, Eigen::Dynamic>, class VectorD = Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1>>
struct iheartmesh {
    std::vector<int > V;
    std::vector<int > E;
    std::vector<int > F;
    std::vector<int > C;
    DT E2;
    DT e;
    Eigen::VectorXd G;
    MatrixD H;
    VectorD d;
    std::vector<Eigen::Matrix<DT, 3, 1>> y;
    Tetrahedron M;
    std::vector<Eigen::Matrix<double, 3, 1>> x̄;
    std::vector<Eigen::Matrix<DT, 3, 1>> x;
    autodiff::ArrayXvar new_x;
    DT vol(
        const int & i,
        const int & j,
        const int & k,
        const int & l)
    {
        assert( std::binary_search(V.begin(), V.end(), i) );
        assert( std::binary_search(V.begin(), V.end(), j) );
        assert( std::binary_search(V.begin(), V.end(), k) );
        assert( std::binary_search(V.begin(), V.end(), l) );

        Eigen::Matrix<DT, 3, 3> vol_0;
        vol_0 << this->x̄.at(j) - this->x̄.at(i), this->x̄.at(k) - this->x̄.at(i), this->x̄.at(l) - this->x̄.at(i);
        return (1/DT(6)) * (vol_0).determinant();    
    }
    MatrixD psd(
        const MatrixD & x)
    {
        const long p = x.cols();
        assert( x.rows() == p );

        Eigen::BDCSVD<Eigen::MatrixXd> svd(to_double(x), Eigen::ComputeFullU | Eigen::ComputeFullV);
        // u, sigma, v = svd(x)
        std::tuple< MatrixD, VectorD, MatrixD > rhs_1 = std::tuple< MatrixD, VectorD, MatrixD >(svd.matrixU(), svd.singularValues(), svd.matrixV());
        MatrixD u = std::get<0>(rhs_1);
        VectorD sigma = std::get<1>(rhs_1);
        MatrixD v = std::get<2>(rhs_1);

        // ps_i = { sigma_i if sigma_i > 0
        //   0 otherwise
        VectorD ps;
        ps.resize(p);
        for( int i=1; i<=p; i++){
            if(sigma[i-1] > 0){
                ps[i-1] = sigma[i-1];
            }
            else{
                ps[i-1] = 0;
            }
        }
        return u * (ps).asDiagonal() * v.transpose();    
    }
    DT S(
        const int & s,
        const std::vector<Eigen::Matrix<DT, 3, 1>> & x)
    {
        const long dim_2 = x.size();
        assert( std::binary_search(C.begin(), C.end(), s) );

        // a, b, c, d = Vertices(s)
        std::vector<int > rhs_2 = Vertices_2(s);
        int a = rhs_2[0];
        int b = rhs_2[1];
        int c = rhs_2[2];
        int d = rhs_2[3];

        // J = [x_b-x_a x_c-x_a x_d-x_a][x̄_b-x̄_a x̄_c-x̄_a x̄_d-x̄_a]⁻¹
        Eigen::Matrix<DT, 3, 3> J_0;
        J_0 << x.at(b) - x.at(a), x.at(c) - x.at(a), x.at(d) - x.at(a);
        Eigen::Matrix<DT, 3, 3> J_1;
        J_1 << this->x̄.at(b) - this->x̄.at(a), this->x̄.at(c) - this->x̄.at(a), this->x̄.at(d) - this->x̄.at(a);
        Eigen::Matrix<DT, 3, 3> J = J_0 * J_1.inverse();
        return vol(a, b, c, d) * (pow((J).norm(), 2) + pow((J.inverse()).norm(), 2));    
    }
        struct TetrahderonNeighborhoods {
        std::vector<int > V;
        std::vector<int > E;
        std::vector<int > F;
        std::vector<int > C;
        Eigen::SparseMatrix<int> dee0;
        Eigen::SparseMatrix<int> dee1;
        Eigen::SparseMatrix<int> dee2;
        Eigen::SparseMatrix<int> B0;
        Eigen::SparseMatrix<int> B1;
        Eigen::SparseMatrix<int> B2;
        Tetrahedron M;
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
            const std::vector<int >& lhs_diff = nonzeros(this->B0 * this->B0.transpose() * M.vertices_to_vector(VertexOneRingset_0));
            const std::vector<int >& rhs_diff = VertexOneRingset_1;
            difference.reserve(lhs_diff.size());
            std::set_difference(lhs_diff.begin(), lhs_diff.end(), rhs_diff.begin(), rhs_diff.end(), std::back_inserter(difference));
            return difference;    
        }
        std::vector<int > VertexOneRing(
            const std::vector<int > & v)
        {
            std::vector<int > difference_1;
            const std::vector<int >& lhs_diff_1 = nonzeros(this->B0 * this->B0.transpose() * M.vertices_to_vector(v));
            const std::vector<int >& rhs_diff_1 = v;
            difference_1.reserve(lhs_diff_1.size());
            std::set_difference(lhs_diff_1.begin(), lhs_diff_1.end(), rhs_diff_1.begin(), rhs_diff_1.end(), std::back_inserter(difference_1));
            return difference_1;    
        }
        int EdgeIndex(
            const int & i,
            const int & j)
        {
            assert( std::binary_search(V.begin(), V.end(), i) );
            assert( std::binary_search(V.begin(), V.end(), j) );
            // eset = edgeset(NonZeros(`∂0`ᵀ IndicatorVector({i}))) ∩ vertexset(NonZeros(`∂0`ᵀ IndicatorVector({j})))
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
            const std::vector<int >& lhs = nonzeros(this->dee0.transpose() * M.vertices_to_vector(EdgeIndexset_0));
            const std::vector<int >& rhs_3 = nonzeros(this->dee0.transpose() * M.vertices_to_vector(EdgeIndexset_1));
            intsect.reserve(std::min(lhs.size(), rhs_3.size()));
            std::set_intersection(lhs.begin(), lhs.end(), rhs_3.begin(), rhs_3.end(), std::back_inserter(intsect));
            std::vector<int > eset = intsect;
            return eset[1-1];    
        }
        std::tuple< int, int > NeighborVerticesInFace(
            const int & f,
            const int & v)
        {
            assert( std::binary_search(V.begin(), V.end(), v) );
            assert( std::binary_search(F.begin(), F.end(), f) );
            // es = edgeset(NonZeros(`∂1` IndicatorVector({f})))
            std::vector<int > NeighborVerticesInFaceset_0({f});
            if(NeighborVerticesInFaceset_0.size() > 1){
                sort(NeighborVerticesInFaceset_0.begin(), NeighborVerticesInFaceset_0.end());
                NeighborVerticesInFaceset_0.erase(unique(NeighborVerticesInFaceset_0.begin(), NeighborVerticesInFaceset_0.end() ), NeighborVerticesInFaceset_0.end());
            }
            std::vector<int > es = nonzeros(this->dee1 * M.faces_to_vector(NeighborVerticesInFaceset_0));
            // nes = { s for s ∈ es if `∂0`_v,s != 0 }
            std::vector<int > NeighborVerticesInFaceset_1;
            const std::vector<int >& range = es;
            NeighborVerticesInFaceset_1.reserve(range.size());
            for(int s : range){
                if(this->dee0.coeff(v, s) != 0){
                    NeighborVerticesInFaceset_1.push_back(s);
                }
            }
            if(NeighborVerticesInFaceset_1.size() > 1){
                sort(NeighborVerticesInFaceset_1.begin(), NeighborVerticesInFaceset_1.end());
                NeighborVerticesInFaceset_1.erase(unique(NeighborVerticesInFaceset_1.begin(), NeighborVerticesInFaceset_1.end() ), NeighborVerticesInFaceset_1.end());
            }
            std::vector<int > nes = NeighborVerticesInFaceset_1;
            // eset1 = { e for e ∈ nes if `∂1`_e,f `∂0`_v,e = -1}
            std::vector<int > NeighborVerticesInFaceset_2;
            const std::vector<int >& range_1 = nes;
            NeighborVerticesInFaceset_2.reserve(range_1.size());
            for(int e : range_1){
                if(this->dee1.coeff(e, f) * this->dee0.coeff(v, e) == -1){
                    NeighborVerticesInFaceset_2.push_back(e);
                }
            }
            if(NeighborVerticesInFaceset_2.size() > 1){
                sort(NeighborVerticesInFaceset_2.begin(), NeighborVerticesInFaceset_2.end());
                NeighborVerticesInFaceset_2.erase(unique(NeighborVerticesInFaceset_2.begin(), NeighborVerticesInFaceset_2.end() ), NeighborVerticesInFaceset_2.end());
            }
            std::vector<int > eset1 = NeighborVerticesInFaceset_2;
            // vset1 = vertexset(NonZeros(B0 IndicatorVector(eset1))) - {v}
            std::vector<int > NeighborVerticesInFaceset_3({v});
            if(NeighborVerticesInFaceset_3.size() > 1){
                sort(NeighborVerticesInFaceset_3.begin(), NeighborVerticesInFaceset_3.end());
                NeighborVerticesInFaceset_3.erase(unique(NeighborVerticesInFaceset_3.begin(), NeighborVerticesInFaceset_3.end() ), NeighborVerticesInFaceset_3.end());
            }
            std::vector<int > difference_2;
            const std::vector<int >& lhs_diff_2 = nonzeros(this->B0 * M.edges_to_vector(eset1));
            const std::vector<int >& rhs_diff_2 = NeighborVerticesInFaceset_3;
            difference_2.reserve(lhs_diff_2.size());
            std::set_difference(lhs_diff_2.begin(), lhs_diff_2.end(), rhs_diff_2.begin(), rhs_diff_2.end(), std::back_inserter(difference_2));
            std::vector<int > vset1 = difference_2;
            // eset2 = { e for e ∈ nes if `∂1`_e,f `∂0`_v,e = 1 }
            std::vector<int > NeighborVerticesInFaceset_4;
            const std::vector<int >& range_2 = nes;
            NeighborVerticesInFaceset_4.reserve(range_2.size());
            for(int e : range_2){
                if(this->dee1.coeff(e, f) * this->dee0.coeff(v, e) == 1){
                    NeighborVerticesInFaceset_4.push_back(e);
                }
            }
            if(NeighborVerticesInFaceset_4.size() > 1){
                sort(NeighborVerticesInFaceset_4.begin(), NeighborVerticesInFaceset_4.end());
                NeighborVerticesInFaceset_4.erase(unique(NeighborVerticesInFaceset_4.begin(), NeighborVerticesInFaceset_4.end() ), NeighborVerticesInFaceset_4.end());
            }
            std::vector<int > eset2 = NeighborVerticesInFaceset_4;
            // vset2 = vertexset(NonZeros(B0 IndicatorVector(eset2))) - {v}
            std::vector<int > NeighborVerticesInFaceset_5({v});
            if(NeighborVerticesInFaceset_5.size() > 1){
                sort(NeighborVerticesInFaceset_5.begin(), NeighborVerticesInFaceset_5.end());
                NeighborVerticesInFaceset_5.erase(unique(NeighborVerticesInFaceset_5.begin(), NeighborVerticesInFaceset_5.end() ), NeighborVerticesInFaceset_5.end());
            }
            std::vector<int > difference_3;
            const std::vector<int >& lhs_diff_3 = nonzeros(this->B0 * M.edges_to_vector(eset2));
            const std::vector<int >& rhs_diff_3 = NeighborVerticesInFaceset_5;
            difference_3.reserve(lhs_diff_3.size());
            std::set_difference(lhs_diff_3.begin(), lhs_diff_3.end(), rhs_diff_3.begin(), rhs_diff_3.end(), std::back_inserter(difference_3));
            std::vector<int > vset2 = difference_3;
            return std::tuple<int,int >{ vset1[1-1],vset2[1-1] };    
        }
        int NextVerticeInFace(
            const int & f,
            const int & v)
        {
            assert( std::binary_search(V.begin(), V.end(), v) );
            assert( std::binary_search(F.begin(), F.end(), f) );
            // eset = { e for e ∈ Edges(f) if `∂1`_e,f `∂0`_v,e = -1}
            std::vector<int > NextVerticeInFaceset_0;
            const std::vector<int >& range_3 = this->Edges(f);
            NextVerticeInFaceset_0.reserve(range_3.size());
            for(int e : range_3){
                if(this->dee1.coeff(e, f) * this->dee0.coeff(v, e) == -1){
                    NextVerticeInFaceset_0.push_back(e);
                }
            }
            if(NextVerticeInFaceset_0.size() > 1){
                sort(NextVerticeInFaceset_0.begin(), NextVerticeInFaceset_0.end());
                NextVerticeInFaceset_0.erase(unique(NextVerticeInFaceset_0.begin(), NextVerticeInFaceset_0.end() ), NextVerticeInFaceset_0.end());
            }
            std::vector<int > eset = NextVerticeInFaceset_0;
            // vset = Vertices(eset) - {v}
            std::vector<int > NextVerticeInFaceset_1({v});
            if(NextVerticeInFaceset_1.size() > 1){
                sort(NextVerticeInFaceset_1.begin(), NextVerticeInFaceset_1.end());
                NextVerticeInFaceset_1.erase(unique(NextVerticeInFaceset_1.begin(), NextVerticeInFaceset_1.end() ), NextVerticeInFaceset_1.end());
            }
            std::vector<int > difference_4;
            const std::vector<int >& lhs_diff_4 = this->Vertices(eset);
            const std::vector<int >& rhs_diff_4 = NextVerticeInFaceset_1;
            difference_4.reserve(lhs_diff_4.size());
            std::set_difference(lhs_diff_4.begin(), lhs_diff_4.end(), rhs_diff_4.begin(), rhs_diff_4.end(), std::back_inserter(difference_4));
            std::vector<int > vset = difference_4;
            return vset[1-1];    
        }
        std::tuple< int, int, int > OrientedVertices(
            const int & f)
        {
            assert( std::binary_search(F.begin(), F.end(), f) );
            // vs = Vertices(f)
            std::vector<int > vs = this->Vertices(f);
            // i,j = NeighborVerticesInFace(f, vs_1)
            std::tuple< int, int > rhs_4 = NeighborVerticesInFace(f, vs[1-1]);
            int i = std::get<0>(rhs_4);
            int j = std::get<1>(rhs_4);
            return std::tuple<int,int,int >{ vs[1-1],i,j };    
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
            return std::tuple<std::vector<int >,std::vector<int >,std::vector<int >,std::vector<int > >{ this->Vertices(e),Diamondset_0,this->Faces(e),tetset };    
        }
        std::tuple< int, int > OppositeVertices(
            const int & e)
        {
            assert( std::binary_search(E.begin(), E.end(), e) );
            // eset = Vertices(Faces(e)) \ Vertices(e)
            std::vector<int > difference_5;
            const std::vector<int >& lhs_diff_5 = this->Vertices(this->Faces(e));
            const std::vector<int >& rhs_diff_5 = this->Vertices(e);
            difference_5.reserve(lhs_diff_5.size());
            std::set_difference(lhs_diff_5.begin(), lhs_diff_5.end(), rhs_diff_5.begin(), rhs_diff_5.end(), std::back_inserter(difference_5));
            std::vector<int > eset = difference_5;
            return std::tuple<int,int >{ eset[1-1],eset[2-1] };    
        }
        int FaceIndex(
            const int & i,
            const int & j,
            const int & k)
        {
            assert( std::binary_search(V.begin(), V.end(), i) );
            assert( std::binary_search(V.begin(), V.end(), j) );
            assert( std::binary_search(V.begin(), V.end(), k) );
            // ufv = (B0 B1)ᵀ
            Eigen::SparseMatrix<int> ufv = (this->B0 * this->B1).transpose();
            // iface = faceset(NonZeros(ufv  IndicatorVector({i})))
            std::vector<int > FaceIndexset_0({i});
            if(FaceIndexset_0.size() > 1){
                sort(FaceIndexset_0.begin(), FaceIndexset_0.end());
                FaceIndexset_0.erase(unique(FaceIndexset_0.begin(), FaceIndexset_0.end() ), FaceIndexset_0.end());
            }
            std::vector<int > iface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_0));
            // jface = faceset(NonZeros(ufv  IndicatorVector({j})))
            std::vector<int > FaceIndexset_1({j});
            if(FaceIndexset_1.size() > 1){
                sort(FaceIndexset_1.begin(), FaceIndexset_1.end());
                FaceIndexset_1.erase(unique(FaceIndexset_1.begin(), FaceIndexset_1.end() ), FaceIndexset_1.end());
            }
            std::vector<int > jface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_1));
            // V, E, F, C = ElementSets( M )0
            std::vector<int > FaceIndexset_2({k});
            if(FaceIndexset_2.size() > 1){
                sort(FaceIndexset_2.begin(), FaceIndexset_2.end());
                FaceIndexset_2.erase(unique(FaceIndexset_2.begin(), FaceIndexset_2.end() ), FaceIndexset_2.end());
            }
            std::vector<int > kface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_2));
            // V, E, F, C = ElementSets( M )1
            std::vector<int > intsect_1;
            const std::vector<int >& lhs_1 = jface;
            const std::vector<int >& rhs_5 = kface;
            intsect_1.reserve(std::min(lhs_1.size(), rhs_5.size()));
            std::set_intersection(lhs_1.begin(), lhs_1.end(), rhs_5.begin(), rhs_5.end(), std::back_inserter(intsect_1));
            std::vector<int > intsect_2;
            const std::vector<int >& lhs_2 = iface;
            const std::vector<int >& rhs_6 = intsect_1;
            intsect_2.reserve(std::min(lhs_2.size(), rhs_6.size()));
            std::set_intersection(lhs_2.begin(), lhs_2.end(), rhs_6.begin(), rhs_6.end(), std::back_inserter(intsect_2));
            std::vector<int > fset = intsect_2;
            return fset[1-1];    
        }
        std::tuple< int, int > OrientedVertices(
            const int & i,
            const int & j,
            const int & k)
        {
            assert( std::binary_search(V.begin(), V.end(), i) );
            assert( std::binary_search(V.begin(), V.end(), j) );
            assert( std::binary_search(V.begin(), V.end(), k) );
            // V, E, F, C = ElementSets( M )4
            int f = FaceIndex(i, j, k);
            return NeighborVerticesInFace(f, i);    
        }
            struct FundamentalTetrahedronAccessors {
            std::vector<int > V;
            std::vector<int > E;
            std::vector<int > F;
            std::vector<int > C;
            Eigen::SparseMatrix<int> dee0;
            Eigen::SparseMatrix<int> dee1;
            Eigen::SparseMatrix<int> dee2;
            Eigen::SparseMatrix<int> B0;
            Eigen::SparseMatrix<int> B1;
            Eigen::SparseMatrix<int> B2;
            Tetrahedron M;
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
            std::vector<int > Tets(
                const std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > & S)
            {
                return std::get<4-1>(S);    
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
                return nonzeros(this->B0 * this->B1 * M.faces_to_vector(Vertices_0set_0));    
            }
            std::vector<int > Vertices_1(
                const std::vector<int > & G)
            {
                return nonzeros(this->B0 * this->B1 * M.faces_to_vector(G));    
            }
            std::vector<int > Vertices_2(
                const int & t)
            {
                assert( std::binary_search(C.begin(), C.end(), t) );
                std::vector<int > Vertices_2set_0({t});
                if(Vertices_2set_0.size() > 1){
                    sort(Vertices_2set_0.begin(), Vertices_2set_0.end());
                    Vertices_2set_0.erase(unique(Vertices_2set_0.begin(), Vertices_2set_0.end() ), Vertices_2set_0.end());
                }
                return nonzeros(this->B0 * this->B1 * this->B2 * M.tets_to_vector(Vertices_2set_0));    
            }
            std::vector<int > Vertices_3(
                const std::vector<int > & J)
            {
                return nonzeros(this->B0 * this->B1 * this->B2 * M.tets_to_vector(J));    
            }
            std::vector<int > Vertices_4(
                const int & e)
            {
                assert( std::binary_search(E.begin(), E.end(), e) );
                std::vector<int > Vertices_4set_0({e});
                if(Vertices_4set_0.size() > 1){
                    sort(Vertices_4set_0.begin(), Vertices_4set_0.end());
                    Vertices_4set_0.erase(unique(Vertices_4set_0.begin(), Vertices_4set_0.end() ), Vertices_4set_0.end());
                }
                return nonzeros(this->B0 * M.edges_to_vector(Vertices_4set_0));    
            }
            std::vector<int > Vertices_5(
                const std::vector<int > & H)
            {
                return nonzeros(this->B0 * M.edges_to_vector(H));    
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
                return nonzeros(this->B0.transpose() * M.vertices_to_vector(Edges_0set_0));    
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
                return nonzeros(this->B1 * M.faces_to_vector(Edges_1set_0));    
            }
            std::vector<int > Edges_2(
                const int & t)
            {
                assert( std::binary_search(C.begin(), C.end(), t) );
                std::vector<int > Edges_2set_0({t});
                if(Edges_2set_0.size() > 1){
                    sort(Edges_2set_0.begin(), Edges_2set_0.end());
                    Edges_2set_0.erase(unique(Edges_2set_0.begin(), Edges_2set_0.end() ), Edges_2set_0.end());
                }
                return nonzeros(this->B1 * this->B2 * M.tets_to_vector(Edges_2set_0));    
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
                return nonzeros((this->B0 * this->B1).transpose() * M.vertices_to_vector(Faces_0set_0));    
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
                return nonzeros(this->B1.transpose() * M.edges_to_vector(Faces_1set_0));    
            }
            std::vector<int > Faces_2(
                const int & t)
            {
                assert( std::binary_search(C.begin(), C.end(), t) );
                std::vector<int > Faces_2set_0({t});
                if(Faces_2set_0.size() > 1){
                    sort(Faces_2set_0.begin(), Faces_2set_0.end());
                    Faces_2set_0.erase(unique(Faces_2set_0.begin(), Faces_2set_0.end() ), Faces_2set_0.end());
                }
                return nonzeros(this->B2 * M.tets_to_vector(Faces_2set_0));    
            }
            FundamentalTetrahedronAccessors(const Tetrahedron & M)
            {
                // V, E, F, C = ElementSets( M )
                std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > rhs = M.ElementSets();
                V = std::get<0>(rhs);
                E = std::get<1>(rhs);
                F = std::get<2>(rhs);
                C = std::get<3>(rhs);
                int dimv_0 = M.n_vertices();
                int dime_0 = M.n_edges();
                int dimf_0 = M.n_faces();
                int dimt_0 = M.n_tets();
                this->M = M;
                // dee0, dee1, dee2 = BoundaryMatrices(M)
                std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_1 = M.BoundaryMatrices();
                dee0 = std::get<0>(rhs_1);
                dee1 = std::get<1>(rhs_1);
                dee2 = std::get<2>(rhs_1);
                // B0, B1, B2 = UnsignedBoundaryMatrices(M)
                std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_2 = M.UnsignedBoundaryMatrices();
                B0 = std::get<0>(rhs_2);
                B1 = std::get<1>(rhs_2);
                B2 = std::get<2>(rhs_2);
            }
        };
        FundamentalTetrahedronAccessors _FundamentalTetrahedronAccessors;
        std::vector<int > Vertices(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
            return _FundamentalTetrahedronAccessors.Vertices(p0);
        };
        std::vector<int > Vertices_0(int p0){
            return _FundamentalTetrahedronAccessors.Vertices_0(p0);
        };
        std::vector<int > Vertices_1(std::vector<int > p0){
            return _FundamentalTetrahedronAccessors.Vertices_1(p0);
        };
        std::vector<int > Vertices_2(int p0){
            return _FundamentalTetrahedronAccessors.Vertices_2(p0);
        };
        std::vector<int > Vertices_3(std::vector<int > p0){
            return _FundamentalTetrahedronAccessors.Vertices_3(p0);
        };
        std::vector<int > Vertices_4(int p0){
            return _FundamentalTetrahedronAccessors.Vertices_4(p0);
        };
        std::vector<int > Vertices_5(std::vector<int > p0){
            return _FundamentalTetrahedronAccessors.Vertices_5(p0);
        };
        std::vector<int > Edges(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
            return _FundamentalTetrahedronAccessors.Edges(p0);
        };
        std::vector<int > Edges_0(int p0){
            return _FundamentalTetrahedronAccessors.Edges_0(p0);
        };
        std::vector<int > Edges_1(int p0){
            return _FundamentalTetrahedronAccessors.Edges_1(p0);
        };
        std::vector<int > Edges_2(int p0){
            return _FundamentalTetrahedronAccessors.Edges_2(p0);
        };
        std::vector<int > Faces(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
            return _FundamentalTetrahedronAccessors.Faces(p0);
        };
        std::vector<int > Faces_0(int p0){
            return _FundamentalTetrahedronAccessors.Faces_0(p0);
        };
        std::vector<int > Faces_1(int p0){
            return _FundamentalTetrahedronAccessors.Faces_1(p0);
        };
        std::vector<int > Faces_2(int p0){
            return _FundamentalTetrahedronAccessors.Faces_2(p0);
        };
        TetrahderonNeighborhoods(const Tetrahedron & M)
        :
        _FundamentalTetrahedronAccessors(M)
        {
            // V, E, F, C = ElementSets( M )
            std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > rhs = M.ElementSets();
            V = std::get<0>(rhs);
            E = std::get<1>(rhs);
            F = std::get<2>(rhs);
            C = std::get<3>(rhs);
            int dimv_0 = M.n_vertices();
            int dime_0 = M.n_edges();
            int dimf_0 = M.n_faces();
            int dimt_0 = M.n_tets();
            this->M = M;
            // `∂0`, `∂1`, `∂2` = BoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_1 = M.BoundaryMatrices();
            dee0 = std::get<0>(rhs_1);
            dee1 = std::get<1>(rhs_1);
            dee2 = std::get<2>(rhs_1);
            // B0, B1, B2 = UnsignedBoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_2 = M.UnsignedBoundaryMatrices();
            B0 = std::get<0>(rhs_2);
            B1 = std::get<1>(rhs_2);
            B2 = std::get<2>(rhs_2);
        }
    };
    TetrahderonNeighborhoods _TetrahderonNeighborhoods;
    std::tuple< int, int > NeighborVerticesInFace(int p0,int p1){
        return _TetrahderonNeighborhoods.NeighborVerticesInFace(p0,p1);
    };
    std::vector<int > Faces(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
        return _TetrahderonNeighborhoods.Faces(p0);
    };
    std::vector<int > Faces_0(int p0){
        return _TetrahderonNeighborhoods.Faces_0(p0);
    };
    std::vector<int > Faces_1(int p0){
        return _TetrahderonNeighborhoods.Faces_1(p0);
    };
    std::vector<int > Faces_2(int p0){
        return _TetrahderonNeighborhoods.Faces_2(p0);
    };
    std::vector<int > Vertices(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
        return _TetrahderonNeighborhoods.Vertices(p0);
    };
    std::vector<int > Vertices_0(int p0){
        return _TetrahderonNeighborhoods.Vertices_0(p0);
    };
    std::vector<int > Vertices_1(std::vector<int > p0){
        return _TetrahderonNeighborhoods.Vertices_1(p0);
    };
    std::vector<int > Vertices_2(int p0){
        return _TetrahderonNeighborhoods.Vertices_2(p0);
    };
    std::vector<int > Vertices_3(std::vector<int > p0){
        return _TetrahderonNeighborhoods.Vertices_3(p0);
    };
    std::vector<int > Vertices_4(int p0){
        return _TetrahderonNeighborhoods.Vertices_4(p0);
    };
    std::vector<int > Vertices_5(std::vector<int > p0){
        return _TetrahderonNeighborhoods.Vertices_5(p0);
    };
    std::vector<int > VertexOneRing(int p0){
        return _TetrahderonNeighborhoods.VertexOneRing(p0);
    };
    std::vector<int > VertexOneRing(std::vector<int > p0){
        return _TetrahderonNeighborhoods.VertexOneRing(p0);
    };
    std::tuple< int, int, int > OrientedVertices(int p0){
        return _TetrahderonNeighborhoods.OrientedVertices(p0);
    };
    std::tuple< int, int > OrientedVertices(int p0,int p1,int p2){
        return _TetrahderonNeighborhoods.OrientedVertices(p0,p1,p2);
    };
    iheartmesh(
        const Tetrahedron & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x̄,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x,
        const std::vector<int> & bx,
        const std::vector<Eigen::Matrix<double, 3, 1>> & bp,
        const double & w,
        const double & ε)
    :
    _TetrahderonNeighborhoods(M)
    {
        // V, E, F, C = ElementSets(M)
        std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > rhs = M.ElementSets();
        V = std::get<0>(rhs);
        E = std::get<1>(rhs);
        F = std::get<2>(rhs);
        C = std::get<3>(rhs);
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        int dimf_0 = M.n_faces();
        int dimt_0 = M.n_tets();
        const long dim_0 = bx.size();
        const long dim_1 = x̄.size();
        assert( x.size() == dim_1 );
        assert( bp.size() == dim_0 );
        this->M = M;
        this->x̄ = x̄;
        new_x.resize(dim_1*3);
        for (int i = 0; i < x.size(); ++i)
        {
            new_x.segment(3*i, 3) = x[i];
        }
        this->x.resize(x.size());
        for (int i = 0; i < x.size(); ++i)
        {
            this->x[i] = new_x.segment(3*i, 3);
        }
        // E2 = w sum_j ‖bp_j - x_(bx_j)‖
        DT sum_0 = 0;
        for(int j=1; j<=bp.size(); j++){
            sum_0 += (bp.at(j-1) - this->x.at(bx.at(j-1))).template lpNorm<2>();
        }
        E2 = w * sum_0;
        // e = sum_(i ∈ C) S(i, x) + E2
        DT sum_1 = 0;
        for(int i : this->C){
            sum_1 += S(i, this->x);
        }
        e = sum_1 + E2;
        // G = ∂e/∂x
        G = gradient(e, this->new_x);
        // H = sum_(i ∈ C) psd(∂²S(i, x)/∂x²) + ∂²E2/∂x²
        MatrixD sum_2 = MatrixD::Zero(3*dim_1, 3*dim_1);
        for(int i : this->C){
            sum_2 += psd(hessian(S(i, this->x), this->new_x).sparseView());
        }
        H = sum_2 + psd(hessian(E2, this->new_x).sparseView());
        // d = H⁻¹ (-G)
        d = (to_double(H)).colPivHouseholderQr().solve(to_double((-G)));
        // y = { inversevec(vec(x) + 0.1 d, x) if √(-d⋅G) > ε
        //       x otherwise
        var eep = sqrt((-(d).dot(G)));
        std::cout<<"eep: "<<eep<<std::endl;
        if(eep > ε){
            std::cout<<"updated"<<std::endl;
            VectorD vec(dim_1*3);
            for (int i = 0; i < this->x.size(); ++i)
            {
                vec.segment(3*i, 3) = this->x[i];
            }
            std::vector<Eigen::Matrix<DT, 3, 1>> inversevec(dim_1);
            VectorD param = vec + 1e-16 * d;
            for (int i = 0; i < this->x.size(); ++i)
            {
                inversevec[i] = param.segment(3*i, 3);
            }
            y = inversevec;
            std::cout<<"y0: "<<y[0]<<std::endl;
        }
        else{
            std::cout<<"the same"<<std::endl;
            y = this->x;
        }
    }
};
 