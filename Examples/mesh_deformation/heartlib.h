/*
vec, inversevec, diag, svd from linearalgebra
ElementSets from MeshConnectivity
NeighborVerticesInFace, Faces, Vertices, VertexOneRing, OrientedVertices from TetrahderonNeighborhoods(M)
M : CellMesh
x̄_i ∈ ℝ³ : rest pos
x_i ∈ ℝ³ : current pos
bx_j ∈ ℤ index: boundary indices
bp_j ∈ ℝ³ : boundary positions
w ∈ ℝ : penalty
ε ∈ ℝ : eps
psd : ℝ^(p×p) -> ℝ^(p×p) sparse

V, E, F, C = ElementSets(M)

vol_i,j,k,l = ⅙ |[x̄_j-x̄_i x̄_k-x̄_i x̄_l-x̄_i]| where i,j,k,l ∈ V

`m_r`(s) = [x̄_b-x̄_a x̄_c-x̄_a x̄_d-x̄_a] where s ∈ C,
a, b, c, d = OrientedVertices(s)

S(s, x) = { ∞ if |m| ≤ 0
            vol_abcd (‖J‖² + ‖J⁻¹‖²) otherwise 
where s ∈ C, x_i ∈ ℝ³,
a, b, c, d = OrientedVertices(s),
m = [x_b-x_a x_c-x_a x_d-x_a],
J = m `m_r`(s)⁻¹

EXPS(s, x) = { ∞ if |m| ≤ 0
               vol_abcd exp(‖J‖² + ‖J⁻¹‖²) otherwise 
where s ∈ C, x_i ∈ ℝ³,
a, b, c, d = OrientedVertices(s),
m = [x_b-x_a x_c-x_a x_d-x_a],
J = m `m_r`(s)⁻¹

AMIPS(s, x) = { ∞ if |m| ≤ 0
                vol_abcd exp(½(‖J‖²/|J| + ½(|J|+|J⁻¹|))) otherwise 
where s ∈ C, x_i ∈ ℝ³,
a, b, c, d = OrientedVertices(s),
m = [x_b-x_a x_c-x_a x_d-x_a],
J = m `m_r`(s)⁻¹

CAMIPS(s, x) = { ∞ if |m| ≤ 0
                 vol_abcd (‖J‖²/|J|^⅔) otherwise 
where s ∈ C, x_i ∈ ℝ³,
a, b, c, d = OrientedVertices(s),
m = [x_b-x_a x_c-x_a x_d-x_a],
J = m `m_r`(s)⁻¹

E2 = w ∑_j ‖bp_j - x_(bx_j)‖²
e = ∑_(i ∈ C) S_i(x) + E2
G = ∂e/∂x
H = ∑_(i ∈ C) psd(∂²S_i(x)/∂x²) + psd(∂²E2/∂x²)
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
#include "CellMesh.h"

using namespace iheartmesh;
struct heartlib {
    using DT = autodiff::var;
    using MatrixD = Eigen::Matrix<autodiff::var, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorD = Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1>;
    std::vector<int > V;
    std::vector<int > E;
    std::vector<int > F;
    std::vector<int > C;
    DT E2;
    DT e;
    Eigen::VectorXd G;
    Eigen::SparseMatrix<double> H;
    CellMesh M;
    std::vector<Eigen::Matrix<double, 3, 1>> x̄;
    std::vector<Eigen::Matrix<DT, 3, 1>> x;
    autodiff::ArrayXvar new_x;
    double vol(
        const int & i,
        const int & j,
        const int & k,
        const int & l)
    {
        assert( std::binary_search(V.begin(), V.end(), i) );
        assert( std::binary_search(V.begin(), V.end(), j) );
        assert( std::binary_search(V.begin(), V.end(), k) );
        assert( std::binary_search(V.begin(), V.end(), l) );

        Eigen::Matrix<double, 3, 3> vol_0;
        vol_0 << this->x̄.at(j) - this->x̄.at(i), this->x̄.at(k) - this->x̄.at(i), this->x̄.at(l) - this->x̄.at(i);
        return (1/double(6)) * (vol_0).determinant();    
    }
    Eigen::Matrix<double, 3, 3> m_r(
        const int & s)
    {
        assert( std::binary_search(C.begin(), C.end(), s) );

        // a, b, c, d = OrientedVertices(s)
        std::tuple< int, int, int, int > rhs_1 = OrientedVertices_0(s);
        int a = std::get<0>(rhs_1);
        int b = std::get<1>(rhs_1);
        int c = std::get<2>(rhs_1);
        int d = std::get<3>(rhs_1);
        Eigen::Matrix<double, 3, 3> m_r_0;
        m_r_0 << this->x̄.at(b) - this->x̄.at(a), this->x̄.at(c) - this->x̄.at(a), this->x̄.at(d) - this->x̄.at(a);
        return m_r_0;    
    }
    template<typename REAL>
    REAL S(
        const int & s,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & x)
    {
        const long dim_2 = x.size();
        assert( std::binary_search(C.begin(), C.end(), s) );

        REAL S_ret;
        // a, b, c, d = OrientedVertices(s)
        std::tuple< int, int, int, int > rhs_2 = OrientedVertices_0(s);
        int a = std::get<0>(rhs_2);
        int b = std::get<1>(rhs_2);
        int c = std::get<2>(rhs_2);
        int d = std::get<3>(rhs_2);

        // m = [x_b-x_a x_c-x_a x_d-x_a]
        Eigen::Matrix<REAL, 3, 3> m_0;
        m_0 << x.at(b) - x.at(a), x.at(c) - x.at(a), x.at(d) - x.at(a);
        Eigen::Matrix<REAL, 3, 3> m = m_0;

        // J = m `m_r`(s)⁻¹
        Eigen::Matrix<REAL, 3, 3> J = m * m_r(s).inverse();
        if((m).determinant() <= 0){
            S_ret = INFINITY;
        }
        else{
            S_ret = vol(a, b, c, d) * (pow((J).norm(), 2) + pow((J.inverse()).norm(), 2));
        }
        return S_ret;    
    }
    template<typename REAL>
    REAL EXPS(
        const int & s,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & x)
    {
        const long dim_3 = x.size();
        assert( std::binary_search(C.begin(), C.end(), s) );

        REAL EXPS_ret;
        // a, b, c, d = OrientedVertices(s)
        std::tuple< int, int, int, int > rhs_3 = OrientedVertices_0(s);
        int a = std::get<0>(rhs_3);
        int b = std::get<1>(rhs_3);
        int c = std::get<2>(rhs_3);
        int d = std::get<3>(rhs_3);

        // m = [x_b-x_a x_c-x_a x_d-x_a]
        Eigen::Matrix<REAL, 3, 3> m_1;
        m_1 << x.at(b) - x.at(a), x.at(c) - x.at(a), x.at(d) - x.at(a);
        Eigen::Matrix<REAL, 3, 3> m = m_1;

        // J = m `m_r`(s)⁻¹
        Eigen::Matrix<REAL, 3, 3> J = m * m_r(s).inverse();
        if((m).determinant() <= 0){
            EXPS_ret = INFINITY;
        }
        else{
            EXPS_ret = vol(a, b, c, d) * exp(pow((J).norm(), 2) + pow((J.inverse()).norm(), 2));
        }
        return EXPS_ret;    
    }
    template<typename REAL>
    REAL AMIPS(
        const int & s,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & x)
    {
        const long dim_4 = x.size();
        assert( std::binary_search(C.begin(), C.end(), s) );

        REAL AMIPS_ret;
        // a, b, c, d = OrientedVertices(s)
        std::tuple< int, int, int, int > rhs_4 = OrientedVertices_0(s);
        int a = std::get<0>(rhs_4);
        int b = std::get<1>(rhs_4);
        int c = std::get<2>(rhs_4);
        int d = std::get<3>(rhs_4);

        // m = [x_b-x_a x_c-x_a x_d-x_a]
        Eigen::Matrix<REAL, 3, 3> m_2;
        m_2 << x.at(b) - x.at(a), x.at(c) - x.at(a), x.at(d) - x.at(a);
        Eigen::Matrix<REAL, 3, 3> m = m_2;

        // J = m `m_r`(s)⁻¹
        Eigen::Matrix<REAL, 3, 3> J = m * m_r(s).inverse();
        if((m).determinant() <= 0){
            AMIPS_ret = INFINITY;
        }
        else{
            AMIPS_ret = vol(a, b, c, d) * exp((1/double(2)) * (pow((J).norm(), 2) / REAL((J).determinant()) + (1/double(2)) * ((J).determinant() + (J.inverse()).determinant())));
        }
        return AMIPS_ret;    
    }
    template<typename REAL>
    REAL CAMIPS(
        const int & s,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & x)
    {
        const long dim_5 = x.size();
        assert( std::binary_search(C.begin(), C.end(), s) );

        REAL CAMIPS_ret;
        // a, b, c, d = OrientedVertices(s)
        std::tuple< int, int, int, int > rhs_5 = OrientedVertices_0(s);
        int a = std::get<0>(rhs_5);
        int b = std::get<1>(rhs_5);
        int c = std::get<2>(rhs_5);
        int d = std::get<3>(rhs_5);

        // m = [x_b-x_a x_c-x_a x_d-x_a]
        Eigen::Matrix<REAL, 3, 3> m_3;
        m_3 << x.at(b) - x.at(a), x.at(c) - x.at(a), x.at(d) - x.at(a);
        Eigen::Matrix<REAL, 3, 3> m = m_3;

        // J = m `m_r`(s)⁻¹
        Eigen::Matrix<REAL, 3, 3> J = m * m_r(s).inverse();
        if((m).determinant() <= 0){
            CAMIPS_ret = INFINITY;
        }
        else{
            CAMIPS_ret = vol(a, b, c, d) * (pow((J).norm(), 2) / REAL(pow((J).determinant(), (2/double(3)))));
        }
        return CAMIPS_ret;    
    }
    struct TetrahderonNeighborhoods {
        using DT_ = double;
        using MatrixD_ = Eigen::MatrixXd;
        using VectorD_ = Eigen::VectorXd;
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
        Eigen::SparseMatrix<int> B0T;
        Eigen::SparseMatrix<int> B1T;
        Eigen::SparseMatrix<int> B2T;
        CellMesh M;
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
            const std::vector<int >& lhs_diff = nonzeros(this->B0 * (this->B0T * indicator(VertexOneRingset_0, M.n_vertices())));
            const std::vector<int >& rhs_diff = VertexOneRingset_1;
            difference.reserve(lhs_diff.size());
            std::set_difference(lhs_diff.begin(), lhs_diff.end(), rhs_diff.begin(), rhs_diff.end(), std::back_inserter(difference));
            return difference;    
        }
        std::vector<int > VertexOneRing(
            const std::vector<int > & v)
        {
            std::vector<int > difference_1;
            const std::vector<int >& lhs_diff_1 = nonzeros(this->B0 * (this->B0T * indicator(v, M.n_vertices())));
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
            // eset = Edges(i) ∩ Edges(j)
            std::vector<int > intsect;
            const std::vector<int >& lhs = Edges_0(i);
            const std::vector<int >& rhs_3 = Edges_0(j);
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
            // es = edgeset(NonZeros(`∂¹` IndicatorVector({f})))
            std::vector<int > NeighborVerticesInFaceset_0({f});
            if(NeighborVerticesInFaceset_0.size() > 1){
                sort(NeighborVerticesInFaceset_0.begin(), NeighborVerticesInFaceset_0.end());
                NeighborVerticesInFaceset_0.erase(unique(NeighborVerticesInFaceset_0.begin(), NeighborVerticesInFaceset_0.end() ), NeighborVerticesInFaceset_0.end());
            }
            std::vector<int > es = nonzeros(this->dee1 * indicator(NeighborVerticesInFaceset_0, M.n_faces()));
            // nes = { s for s ∈ es if `∂⁰`_v,s != 0 }
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
            // eset1 = { e for e ∈ nes if `∂¹`_e,f `∂⁰`_v,e = -1}
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
            // vset1 = vertexset(NonZeros(`B⁰` IndicatorVector(eset1))) - {v}
            std::vector<int > NeighborVerticesInFaceset_3({v});
            if(NeighborVerticesInFaceset_3.size() > 1){
                sort(NeighborVerticesInFaceset_3.begin(), NeighborVerticesInFaceset_3.end());
                NeighborVerticesInFaceset_3.erase(unique(NeighborVerticesInFaceset_3.begin(), NeighborVerticesInFaceset_3.end() ), NeighborVerticesInFaceset_3.end());
            }
            std::vector<int > difference_2;
            const std::vector<int >& lhs_diff_2 = nonzeros(this->B0 * indicator(eset1, M.n_edges()));
            const std::vector<int >& rhs_diff_2 = NeighborVerticesInFaceset_3;
            difference_2.reserve(lhs_diff_2.size());
            std::set_difference(lhs_diff_2.begin(), lhs_diff_2.end(), rhs_diff_2.begin(), rhs_diff_2.end(), std::back_inserter(difference_2));
            std::vector<int > vset1 = difference_2;
            // eset2 = { e for e ∈ nes if `∂¹`_e,f `∂⁰`_v,e = 1 }
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
            // vset2 = vertexset(NonZeros(`B⁰` IndicatorVector(eset2))) - {v}
            std::vector<int > NeighborVerticesInFaceset_5({v});
            if(NeighborVerticesInFaceset_5.size() > 1){
                sort(NeighborVerticesInFaceset_5.begin(), NeighborVerticesInFaceset_5.end());
                NeighborVerticesInFaceset_5.erase(unique(NeighborVerticesInFaceset_5.begin(), NeighborVerticesInFaceset_5.end() ), NeighborVerticesInFaceset_5.end());
            }
            std::vector<int > difference_3;
            const std::vector<int >& lhs_diff_3 = nonzeros(this->B0 * indicator(eset2, M.n_edges()));
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
            // eset = { e for e ∈ Edges(f) if `∂¹`_e,f `∂⁰`_v,e = -1}
            std::vector<int > NextVerticeInFaceset_0;
            const std::vector<int >& range_3 = Edges_1(f);
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
            const std::vector<int >& lhs_diff_4 = Vertices_1(eset);
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
            std::vector<int > vs = Vertices_0(f);
            // i,j = NeighborVerticesInFace(f, vs_1)
            std::tuple< int, int > rhs_4 = NeighborVerticesInFace(f, vs[1-1]);
            int i = std::get<0>(rhs_4);
            int j = std::get<1>(rhs_4);
            return std::tuple<int,int,int >{ vs[1-1],i,j };    
        }
        std::tuple< int, int, int, int > OrientedVertices_0(
            const int & c)
        {
            assert( std::binary_search(C.begin(), C.end(), c) );
            // fset = Faces(c)
            std::vector<int > fset = Faces_2(c);
            // pfset = { f for f ∈ fset if `∂²`_f,c = 1 }
            std::vector<int > OrientedVertices_0set_0;
            const std::vector<int >& range_4 = fset;
            OrientedVertices_0set_0.reserve(range_4.size());
            for(int f : range_4){
                if(this->dee2.coeff(f, c) == 1){
                    OrientedVertices_0set_0.push_back(f);
                }
            }
            if(OrientedVertices_0set_0.size() > 1){
                sort(OrientedVertices_0set_0.begin(), OrientedVertices_0set_0.end());
                OrientedVertices_0set_0.erase(unique(OrientedVertices_0set_0.begin(), OrientedVertices_0set_0.end() ), OrientedVertices_0set_0.end());
            }
            std::vector<int > pfset = OrientedVertices_0set_0;
            // i,j,k = OrientedVertices(pfset_1)
            std::tuple< int, int, int > rhs_5 = OrientedVertices(pfset[1-1]);
            int i = std::get<0>(rhs_5);
            int j = std::get<1>(rhs_5);
            int k = std::get<2>(rhs_5);
            // remain = Vertices(c) - Vertices(pfset_1)
            std::vector<int > difference_5;
            const std::vector<int >& lhs_diff_5 = Vertices_2(c);
            const std::vector<int >& rhs_diff_5 = Vertices_0(pfset[1-1]);
            difference_5.reserve(lhs_diff_5.size());
            std::set_difference(lhs_diff_5.begin(), lhs_diff_5.end(), rhs_diff_5.begin(), rhs_diff_5.end(), std::back_inserter(difference_5));
            std::vector<int > remain = difference_5;
            // l = remain_1
            int l = remain[1-1];
            return std::tuple<int,int,int,int >{ l,i,j,k };    
        }
        int FaceIndex(
            const int & i,
            const int & j,
            const int & k)
        {
            assert( std::binary_search(V.begin(), V.end(), i) );
            assert( std::binary_search(V.begin(), V.end(), j) );
            assert( std::binary_search(V.begin(), V.end(), k) );
            // V, E, F, C = ElementSets( M )1
            Eigen::SparseMatrix<int> ufv = this->B1T * this->B0T;
            // V, E, F, C = ElementSets( M )2
            std::vector<int > FaceIndexset_0({i});
            if(FaceIndexset_0.size() > 1){
                sort(FaceIndexset_0.begin(), FaceIndexset_0.end());
                FaceIndexset_0.erase(unique(FaceIndexset_0.begin(), FaceIndexset_0.end() ), FaceIndexset_0.end());
            }
            std::vector<int > iface = nonzeros(ufv * indicator(FaceIndexset_0, M.n_vertices()));
            // V, E, F, C = ElementSets( M )3
            std::vector<int > FaceIndexset_1({j});
            if(FaceIndexset_1.size() > 1){
                sort(FaceIndexset_1.begin(), FaceIndexset_1.end());
                FaceIndexset_1.erase(unique(FaceIndexset_1.begin(), FaceIndexset_1.end() ), FaceIndexset_1.end());
            }
            std::vector<int > jface = nonzeros(ufv * indicator(FaceIndexset_1, M.n_vertices()));
            // V, E, F, C = ElementSets( M )4
            std::vector<int > FaceIndexset_2({k});
            if(FaceIndexset_2.size() > 1){
                sort(FaceIndexset_2.begin(), FaceIndexset_2.end());
                FaceIndexset_2.erase(unique(FaceIndexset_2.begin(), FaceIndexset_2.end() ), FaceIndexset_2.end());
            }
            std::vector<int > kface = nonzeros(ufv * indicator(FaceIndexset_2, M.n_vertices()));
            // V, E, F, C = ElementSets( M )5
            std::vector<int > intsect_1;
            const std::vector<int >& lhs_1 = jface;
            const std::vector<int >& rhs_6 = kface;
            intsect_1.reserve(std::min(lhs_1.size(), rhs_6.size()));
            std::set_intersection(lhs_1.begin(), lhs_1.end(), rhs_6.begin(), rhs_6.end(), std::back_inserter(intsect_1));
            std::vector<int > intsect_2;
            const std::vector<int >& lhs_2 = iface;
            const std::vector<int >& rhs_7 = intsect_1;
            intsect_2.reserve(std::min(lhs_2.size(), rhs_7.size()));
            std::set_intersection(lhs_2.begin(), lhs_2.end(), rhs_7.begin(), rhs_7.end(), std::back_inserter(intsect_2));
            std::vector<int > fset = intsect_2;
            return fset[1-1];    
        }
        std::tuple< int, int > OrientedVertices_1(
            const int & i,
            const int & j,
            const int & k)
        {
            assert( std::binary_search(V.begin(), V.end(), i) );
            assert( std::binary_search(V.begin(), V.end(), j) );
            assert( std::binary_search(V.begin(), V.end(), k) );
            // V, E, F, C = ElementSets( M )8
            int f = FaceIndex(i, j, k);
            return NeighborVerticesInFace(f, i);    
        }
        struct FundamentalTetrahedronAccessors {
            using DT = double;
            using MatrixD = Eigen::MatrixXd;
            using VectorD = Eigen::VectorXd;
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
            Eigen::SparseMatrix<int> B0T;
            Eigen::SparseMatrix<int> B1T;
            Eigen::SparseMatrix<int> B2T;
            CellMesh M;
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
                return nonzeros(this->B0 * (this->B1 * indicator(Vertices_0set_0, M.n_faces())));    
            }
            std::vector<int > Vertices_1(
                const std::vector<int > & G)
            {
                return nonzeros(this->B0 * (this->B1 * indicator(G, M.n_faces())));    
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
                return nonzeros(this->B0 * (this->B1 * (this->B2 * indicator(Vertices_2set_0, M.n_tets()))));    
            }
            std::vector<int > Vertices_3(
                const std::vector<int > & J)
            {
                return nonzeros(this->B0 * (this->B1 * (this->B2 * indicator(J, M.n_tets()))));    
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
                return nonzeros(this->B0 * indicator(Vertices_4set_0, M.n_edges()));    
            }
            std::vector<int > Vertices_5(
                const std::vector<int > & H)
            {
                return nonzeros(this->B0 * indicator(H, M.n_edges()));    
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
                return nonzeros(this->B0T * indicator(Edges_0set_0, M.n_vertices()));    
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
                return nonzeros(this->B1 * indicator(Edges_1set_0, M.n_faces()));    
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
                return nonzeros(this->B1 * (this->B2 * indicator(Edges_2set_0, M.n_tets())));    
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
                return nonzeros(this->B1T * (this->B0T * indicator(Faces_0set_0, M.n_vertices())));    
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
                return nonzeros(this->B1T * indicator(Faces_1set_0, M.n_edges()));    
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
                return nonzeros(this->B2 * indicator(Faces_2set_0, M.n_tets()));    
            }
            FundamentalTetrahedronAccessors(const CellMesh & M)
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
                // B0T = B0ᵀ
                B0T = B0.transpose();
                // B1T = B1ᵀ
                B1T = B1.transpose();
                // B2T = B2ᵀ
                B2T = B2.transpose();
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
        TetrahderonNeighborhoods(const CellMesh & M)
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
            // `∂⁰`, `∂¹`, `∂²` = BoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_1 = M.BoundaryMatrices();
            dee0 = std::get<0>(rhs_1);
            dee1 = std::get<1>(rhs_1);
            dee2 = std::get<2>(rhs_1);
            // `B⁰`, `B¹`, `B²` = UnsignedBoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_2 = M.UnsignedBoundaryMatrices();
            B0 = std::get<0>(rhs_2);
            B1 = std::get<1>(rhs_2);
            B2 = std::get<2>(rhs_2);
            // `B⁰ᵀ` = `B⁰`ᵀ
            B0T = B0.transpose();
            // `B¹ᵀ` = `B¹`ᵀ
            B1T = B1.transpose();
            // `B²ᵀ` = `B²`ᵀ
            B2T = B2.transpose();
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
    std::tuple< int, int, int, int > OrientedVertices_0(int p0){
        return _TetrahderonNeighborhoods.OrientedVertices_0(p0);
    };
    std::tuple< int, int > OrientedVertices_1(int p0,int p1,int p2){
        return _TetrahderonNeighborhoods.OrientedVertices_1(p0,p1,p2);
    };
    heartlib(
        const CellMesh & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x̄,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x,
        const std::vector<int> & bx,
        const std::vector<Eigen::Matrix<double, 3, 1>> & bp,
        const double & w,
        const double & ε,
        const std::function<Eigen::SparseMatrix<double>(Eigen::MatrixXd)> & psd)
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
        // E2 = w ∑_j ‖bp_j - x_(bx_j)‖²
        DT sum_0 = 0;
        for(int j=1; j<=bp.size(); j++){
            sum_0 += pow((bp.at(j-1) - this->x.at(bx.at(j-1))).template lpNorm<2>(), 2);
        }
        E2 = w * sum_0;
        // e = ∑_(i ∈ C) S_i(x) + E2
        DT sum_1 = 0;
        for(int i : this->C){
            sum_1 += S(i, this->x);
        }
        e = sum_1 + E2;
        // G = ∂e/∂x
        G = gradient(e, this->new_x);
        // H = ∑_(i ∈ C) psd(∂²S_i(x)/∂x²) + psd(∂²E2/∂x²)
        Eigen::SparseMatrix<double> sum_2(3*dim_1, 3*dim_1);
        for(int i : this->C){
            sum_2 += psd(hessian(S(i, this->x), this->new_x));
        }
        H = sum_2 + psd(hessian(E2, this->new_x));
    }
};





