/*
ElementSets from MeshConnectivity
VertexOneRing, Faces, NeighborVerticesInFace from Neighborhoods(M)
M : FaceMesh
x_i ∈ ℝ^3  
V, E, F = ElementSets( M )

UpdateStep(i, j, k, d) = { p if s_1,1 < 0 and s_2,1 < 0 
                           min(d_j+‖x_j - x_i‖, d_k+‖x_k - x_i‖) otherwise 
where i,j,k ∈ V, d_i ∈ ℝ,
X = [x_j-x_i x_k-x_i],
t = [d_j d_k]ᵀ,
Q = (XᵀX)⁻¹,
`1` = [1 ; 1],
p = (`1`ᵀQt + sqrt((`1`ᵀQt)² - `1`ᵀQ`1` ⋅ (tᵀQt - 1)))/ (`1`ᵀQ`1`),
n = XQ(t- p ⋅`1`),
s = QXᵀn

GetNextLevel(U) = v - s where U_i ⊂ V,
s = ∪_i U_i,
v = VertexOneRing(s)

GetRangeLevel(U, a, b) = ∪_(i=a)^b U_i where U_j ⊂ V, a,b ∈ ℤ index


GetLevelSequence(U) = { sequence(U, next) if |next| ≠ 0
                        U otherwise where U_i ⊂ V,
next = GetNextLevel(U)


*/
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include <algorithm>
#include "type_helper.h"
#include "FaceMesh.h"

using namespace iheartmesh;
struct heartlib {
    using DT = double;
    using MatrixD = Eigen::MatrixXd;
    using VectorD = Eigen::VectorXd;
    std::vector<int > V;
    std::vector<int > E;
    std::vector<int > F;
    FaceMesh M;
    std::vector<Eigen::Matrix<double, 3, 1>> x;
    template<typename REAL>
    REAL UpdateStep(
        const int & i,
        const int & j,
        const int & k,
        const std::vector<REAL> & d)
    {
        const long dim_1 = d.size();
        assert( std::binary_search(V.begin(), V.end(), i) );
        assert( std::binary_search(V.begin(), V.end(), j) );
        assert( std::binary_search(V.begin(), V.end(), k) );

        REAL UpdateStep_ret;
        // X = [x_j-x_i x_k-x_i]
        Eigen::Matrix<REAL, 3, 2> X_0;
        X_0 << this->x.at(j) - this->x.at(i), this->x.at(k) - this->x.at(i);
        Eigen::Matrix<REAL, 3, 2> X = X_0;

        // t = [d_j d_k]ᵀ
        Eigen::Matrix<REAL, 1, 2> t_0;
        t_0 << d.at(j), d.at(k);
        Eigen::Matrix<REAL, 2, 1> t = t_0.transpose();

        // Q = (XᵀX)⁻¹
        Eigen::Matrix<REAL, 2, 2> Q = (X.transpose() * X).inverse();

        // `1` = [1 ; 1]
        Eigen::Matrix<int, 2, 1> num1_0;
        num1_0 << 1,
        1;
        Eigen::Matrix<int, 2, 1> num1 = num1_0;

        // p = (`1`ᵀQt + sqrt((`1`ᵀQt)² - `1`ᵀQ`1` ⋅ (tᵀQt - 1)))/ (`1`ᵀQ`1`)
        REAL p = ((REAL)((num1.transpose()).template cast<REAL>() * Q * t) + sqrt(pow(((REAL)((num1.transpose()).template cast<REAL>() * Q * t)), 2) - (REAL)((num1.transpose()).template cast<REAL>() * Q * (num1).template cast<REAL>()) * ((REAL)(t.transpose() * Q * t) - 1))) / REAL(((REAL)((num1.transpose()).template cast<REAL>() * Q * (num1).template cast<REAL>())));

        // n = XQ(t- p ⋅`1`)
        Eigen::Matrix<REAL, 3, 1> n = X * Q * (t - (p * (num1).template cast<REAL>()).template cast<REAL>());

        // s = QXᵀn
        Eigen::Matrix<REAL, 2, 1> s = Q * X.transpose() * n;
        if((s(1-1, 1-1) < 0) && (s(2-1, 1-1) < 0)){
            UpdateStep_ret = p;
        }
        else{
            UpdateStep_ret = std::min({d.at(j) + (this->x.at(j) - this->x.at(i)).template lpNorm<2>(), d.at(k) + (this->x.at(k) - this->x.at(i)).template lpNorm<2>()});
        }
        return UpdateStep_ret;    
    }
    std::vector<int > GetNextLevel(
        const std::vector<std::vector<int >> & U)
    {
        const long dim_2 = U.size();
        // s = ∪_i U_i
        std::vector<int > union_0;
        std::vector<int > tmp;
        for(int i=1; i<=U.size(); i++){
            std::set_union(union_0.begin(), union_0.end(), U.at(i-1).begin(), U.at(i-1).end(), std::back_inserter(tmp));
            union_0.assign(tmp.begin(), tmp.end());
            tmp.clear();
        }
        std::vector<int > s = union_0;

        // v = VertexOneRing(s)
        std::vector<int > v = this->VertexOneRing(s);
        std::vector<int > difference;
        const std::vector<int >& lhs_diff = v;
        const std::vector<int >& rhs_diff = s;
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
        std::vector<int > tmp_1;
        for(int i=a; i<=b; i++){
            std::set_union(union_1.begin(), union_1.end(), U.at(i).begin(), U.at(i).end(), std::back_inserter(tmp_1));
            union_1.assign(tmp_1.begin(), tmp_1.end());
            tmp_1.clear();
        }
        return union_1;    
    }
    std::vector<std::vector<int >> GetLevelSequence(
        const std::vector<std::vector<int >> & U)
    {
        const long dim_4 = U.size();
        std::vector<std::vector<int >> GetLevelSequence_ret;
        // next = GetNextLevel(U)
        std::vector<int > next = GetNextLevel(U);
        if((next).size() != 0){
            std::vector<std::vector<int >> seq = U;
            seq.reserve(seq.size()+1);
            seq.push_back(next);
            GetLevelSequence_ret = seq;
        }
        else{
            GetLevelSequence_ret = U;
        }
        return GetLevelSequence_ret;    
    }
    struct Neighborhoods {
        using DT_ = double;
        using MatrixD_ = Eigen::MatrixXd;
        using VectorD_ = Eigen::VectorXd;
        std::vector<int > V;
        std::vector<int > E;
        std::vector<int > F;
        Eigen::SparseMatrix<int> dee0;
        Eigen::SparseMatrix<int> dee1;
        Eigen::SparseMatrix<int> B0;
        Eigen::SparseMatrix<int> B1;
        Eigen::SparseMatrix<int> B0T;
        Eigen::SparseMatrix<int> B1T;
        FaceMesh M;
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
            // V, E, F = ElementSets( M )0
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
            return std::tuple<std::vector<int >,std::vector<int >,std::vector<int >,std::vector<int > >{ Vertices_2(e),Diamondset_0,Faces_1(e),tetset };    
        }
        std::tuple< int, int > OrientedOppositeFaces(
            const int & i,
            const int & j)
        {
            assert( std::binary_search(V.begin(), V.end(), i) );
            assert( std::binary_search(V.begin(), V.end(), j) );
            // V, E, F = ElementSets( M )5
            int e = EdgeIndex(i, j);
            // V, E, F = ElementSets( M )6
            std::vector<int > fset = Faces_1(e);
            // V, E, F = ElementSets( M )7
            std::vector<int > OrientedOppositeFacesset_0;
            const std::vector<int >& range_4 = fset;
            OrientedOppositeFacesset_0.reserve(range_4.size());
            for(int f : range_4){
                if(this->dee1.coeff(e, f) * this->dee0.coeff(i, e) == -1){
                    OrientedOppositeFacesset_0.push_back(f);
                }
            }
            if(OrientedOppositeFacesset_0.size() > 1){
                sort(OrientedOppositeFacesset_0.begin(), OrientedOppositeFacesset_0.end());
                OrientedOppositeFacesset_0.erase(unique(OrientedOppositeFacesset_0.begin(), OrientedOppositeFacesset_0.end() ), OrientedOppositeFacesset_0.end());
            }
            std::vector<int > firf = OrientedOppositeFacesset_0;
            // V, E, F = ElementSets( M )8
            std::vector<int > difference_5;
            const std::vector<int >& lhs_diff_5 = fset;
            const std::vector<int >& rhs_diff_5 = firf;
            difference_5.reserve(lhs_diff_5.size());
            std::set_difference(lhs_diff_5.begin(), lhs_diff_5.end(), rhs_diff_5.begin(), rhs_diff_5.end(), std::back_inserter(difference_5));
            std::vector<int > secf = difference_5;
            return std::tuple<int,int >{ firf[1-1],secf[1-1] };    
        }
        std::tuple< int, int > OppositeFaces(
            const int & e)
        {
            assert( std::binary_search(E.begin(), E.end(), e) );
            // `∂⁰`, `∂¹` = BoundaryMatrices(M)1
            std::vector<int > fset = Faces_1(e);
            // `∂⁰`, `∂¹` = BoundaryMatrices(M)2
            std::vector<int > OppositeFacesset_0;
            const std::vector<int >& range_5 = fset;
            OppositeFacesset_0.reserve(range_5.size());
            for(int f : range_5){
                if(this->dee1.coeff(e, f) == 1){
                    OppositeFacesset_0.push_back(f);
                }
            }
            if(OppositeFacesset_0.size() > 1){
                sort(OppositeFacesset_0.begin(), OppositeFacesset_0.end());
                OppositeFacesset_0.erase(unique(OppositeFacesset_0.begin(), OppositeFacesset_0.end() ), OppositeFacesset_0.end());
            }
            std::vector<int > firf = OppositeFacesset_0;
            // `∂⁰`, `∂¹` = BoundaryMatrices(M)3
            std::vector<int > OppositeFacesset_1;
            const std::vector<int >& range_6 = fset;
            OppositeFacesset_1.reserve(range_6.size());
            for(int f : range_6){
                if(this->dee1.coeff(e, f) == -1){
                    OppositeFacesset_1.push_back(f);
                }
            }
            if(OppositeFacesset_1.size() > 1){
                sort(OppositeFacesset_1.begin(), OppositeFacesset_1.end());
                OppositeFacesset_1.erase(unique(OppositeFacesset_1.begin(), OppositeFacesset_1.end() ), OppositeFacesset_1.end());
            }
            std::vector<int > secf = OppositeFacesset_1;
            return std::tuple<int,int >{ firf[1-1],secf[1-1] };    
        }
        std::tuple< int, int > OppositeVertices(
            const int & e)
        {
            assert( std::binary_search(E.begin(), E.end(), e) );
            // `∂⁰`, `∂¹` = BoundaryMatrices(M)6
            std::tuple< int, int > rhs_5 = OppositeFaces(e);
            int firf = std::get<0>(rhs_5);
            int secf = std::get<1>(rhs_5);
            // `∂⁰`, `∂¹` = BoundaryMatrices(M)7
            std::vector<int > difference_6;
            const std::vector<int >& lhs_diff_6 = Vertices_0(firf);
            const std::vector<int >& rhs_diff_6 = Vertices_2(e);
            difference_6.reserve(lhs_diff_6.size());
            std::set_difference(lhs_diff_6.begin(), lhs_diff_6.end(), rhs_diff_6.begin(), rhs_diff_6.end(), std::back_inserter(difference_6));
            std::vector<int > firv = difference_6;
            // `∂⁰`, `∂¹` = BoundaryMatrices(M)8
            std::vector<int > difference_7;
            const std::vector<int >& lhs_diff_7 = Vertices_0(secf);
            const std::vector<int >& rhs_diff_7 = Vertices_2(e);
            difference_7.reserve(lhs_diff_7.size());
            std::set_difference(lhs_diff_7.begin(), lhs_diff_7.end(), rhs_diff_7.begin(), rhs_diff_7.end(), std::back_inserter(difference_7));
            std::vector<int > secv = difference_7;
            return std::tuple<int,int >{ firv[1-1],secv[1-1] };    
        }
        std::tuple< int, int > OppositeVertices(
            const int & i,
            const int & j)
        {
            assert( std::binary_search(V.begin(), V.end(), i) );
            assert( std::binary_search(V.begin(), V.end(), j) );
            // `B⁰`, `B¹` = UnsignedBoundaryMatrices(M)1
            std::tuple< int, int > rhs_6 = OrientedOppositeFaces(i, j);
            int firf = std::get<0>(rhs_6);
            int secf = std::get<1>(rhs_6);
            // `B⁰`, `B¹` = UnsignedBoundaryMatrices(M)2
            std::vector<int > OppositeVertices_0set_0({i, j});
            if(OppositeVertices_0set_0.size() > 1){
                sort(OppositeVertices_0set_0.begin(), OppositeVertices_0set_0.end());
                OppositeVertices_0set_0.erase(unique(OppositeVertices_0set_0.begin(), OppositeVertices_0set_0.end() ), OppositeVertices_0set_0.end());
            }
            std::vector<int > difference_8;
            const std::vector<int >& lhs_diff_8 = Vertices_0(firf);
            const std::vector<int >& rhs_diff_8 = OppositeVertices_0set_0;
            difference_8.reserve(lhs_diff_8.size());
            std::set_difference(lhs_diff_8.begin(), lhs_diff_8.end(), rhs_diff_8.begin(), rhs_diff_8.end(), std::back_inserter(difference_8));
            std::vector<int > firv = difference_8;
            // `B⁰`, `B¹` = UnsignedBoundaryMatrices(M)3
            std::vector<int > OppositeVertices_0set_1({i, j});
            if(OppositeVertices_0set_1.size() > 1){
                sort(OppositeVertices_0set_1.begin(), OppositeVertices_0set_1.end());
                OppositeVertices_0set_1.erase(unique(OppositeVertices_0set_1.begin(), OppositeVertices_0set_1.end() ), OppositeVertices_0set_1.end());
            }
            std::vector<int > difference_9;
            const std::vector<int >& lhs_diff_9 = Vertices_0(secf);
            const std::vector<int >& rhs_diff_9 = OppositeVertices_0set_1;
            difference_9.reserve(lhs_diff_9.size());
            std::set_difference(lhs_diff_9.begin(), lhs_diff_9.end(), rhs_diff_9.begin(), rhs_diff_9.end(), std::back_inserter(difference_9));
            std::vector<int > secv = difference_9;
            return std::tuple<int,int >{ firv[1-1],secv[1-1] };    
        }
        int FaceIndex(
            const int & i,
            const int & j,
            const int & k)
        {
            assert( std::binary_search(V.begin(), V.end(), i) );
            assert( std::binary_search(V.begin(), V.end(), j) );
            assert( std::binary_search(V.begin(), V.end(), k) );
            // `B⁰`, `B¹` = UnsignedBoundaryMatrices(M)6
            Eigen::SparseMatrix<int> ufv = this->B1.transpose() * this->B0.transpose();
            // `B⁰`, `B¹` = UnsignedBoundaryMatrices(M)7
            std::vector<int > FaceIndexset_0({i});
            if(FaceIndexset_0.size() > 1){
                sort(FaceIndexset_0.begin(), FaceIndexset_0.end());
                FaceIndexset_0.erase(unique(FaceIndexset_0.begin(), FaceIndexset_0.end() ), FaceIndexset_0.end());
            }
            std::vector<int > iface = nonzeros(ufv * indicator(FaceIndexset_0, M.n_vertices()));
            // `B⁰`, `B¹` = UnsignedBoundaryMatrices(M)8
            std::vector<int > FaceIndexset_1({j});
            if(FaceIndexset_1.size() > 1){
                sort(FaceIndexset_1.begin(), FaceIndexset_1.end());
                FaceIndexset_1.erase(unique(FaceIndexset_1.begin(), FaceIndexset_1.end() ), FaceIndexset_1.end());
            }
            std::vector<int > jface = nonzeros(ufv * indicator(FaceIndexset_1, M.n_vertices()));
            // `B⁰`, `B¹` = UnsignedBoundaryMatrices(M)9
            std::vector<int > FaceIndexset_2({k});
            if(FaceIndexset_2.size() > 1){
                sort(FaceIndexset_2.begin(), FaceIndexset_2.end());
                FaceIndexset_2.erase(unique(FaceIndexset_2.begin(), FaceIndexset_2.end() ), FaceIndexset_2.end());
            }
            std::vector<int > kface = nonzeros(ufv * indicator(FaceIndexset_2, M.n_vertices()));
            // `B⁰ᵀ` = `B⁰`ᵀ0
            std::vector<int > intsect_1;
            const std::vector<int >& lhs_1 = jface;
            const std::vector<int >& rhs_7 = kface;
            intsect_1.reserve(std::min(lhs_1.size(), rhs_7.size()));
            std::set_intersection(lhs_1.begin(), lhs_1.end(), rhs_7.begin(), rhs_7.end(), std::back_inserter(intsect_1));
            std::vector<int > intsect_2;
            const std::vector<int >& lhs_2 = iface;
            const std::vector<int >& rhs_8 = intsect_1;
            intsect_2.reserve(std::min(lhs_2.size(), rhs_8.size()));
            std::set_intersection(lhs_2.begin(), lhs_2.end(), rhs_8.begin(), rhs_8.end(), std::back_inserter(intsect_2));
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
            // `B⁰ᵀ` = `B⁰`ᵀ3
            int f = FaceIndex(i, j, k);
            return NeighborVerticesInFace(f, i);    
        }
        struct FundamentalMeshAccessors {
            using DT = double;
            using MatrixD = Eigen::MatrixXd;
            using VectorD = Eigen::VectorXd;
            std::vector<int > V;
            std::vector<int > E;
            std::vector<int > F;
            Eigen::SparseMatrix<int> dee0;
            Eigen::SparseMatrix<int> dee1;
            Eigen::SparseMatrix<int> B0;
            Eigen::SparseMatrix<int> B1;
            Eigen::SparseMatrix<int> B0T;
            Eigen::SparseMatrix<int> B1T;
            FaceMesh M;
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
                return nonzeros(this->B0 * (this->B1 * indicator(Vertices_0set_0, M.n_faces())));    
            }
            std::vector<int > Vertices_1(
                const std::vector<int > & G)
            {
                return nonzeros(this->B0 * (this->B1 * indicator(G, M.n_faces())));    
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
                return nonzeros(this->B0 * indicator(Vertices_2set_0, M.n_edges()));    
            }
            std::vector<int > Vertices_3(
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
            FundamentalMeshAccessors(const FaceMesh & M)
            {
                // V, E, F = ElementSets( M )
                std::tuple< std::vector<int >, std::vector<int >, std::vector<int > > rhs = M.ElementSets();
                V = std::get<0>(rhs);
                E = std::get<1>(rhs);
                F = std::get<2>(rhs);
                int dimv_0 = M.n_vertices();
                int dime_0 = M.n_edges();
                int dimf_0 = M.n_faces();
                this->M = M;
                // dee0, dee1 = BoundaryMatrices(M)
                std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_1 = M.BoundaryMatrices();
                dee0 = std::get<0>(rhs_1);
                dee1 = std::get<1>(rhs_1);
                // B0, B1 = UnsignedBoundaryMatrices(M)
                std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_2 = M.UnsignedBoundaryMatrices();
                B0 = std::get<0>(rhs_2);
                B1 = std::get<1>(rhs_2);
                // B0T = B0ᵀ
                B0T = B0.transpose();
                // B1T = B1ᵀ
                B1T = B1.transpose();
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
        Neighborhoods(const FaceMesh & M)
        :
        _FundamentalMeshAccessors(M)
        {
            // V, E, F = ElementSets( M )
            std::tuple< std::vector<int >, std::vector<int >, std::vector<int > > rhs = M.ElementSets();
            V = std::get<0>(rhs);
            E = std::get<1>(rhs);
            F = std::get<2>(rhs);
            int dimv_0 = M.n_vertices();
            int dime_0 = M.n_edges();
            int dimf_0 = M.n_faces();
            this->M = M;
            // `∂⁰`, `∂¹` = BoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_1 = M.BoundaryMatrices();
            dee0 = std::get<0>(rhs_1);
            dee1 = std::get<1>(rhs_1);
            // `B⁰`, `B¹` = UnsignedBoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_2 = M.UnsignedBoundaryMatrices();
            B0 = std::get<0>(rhs_2);
            B1 = std::get<1>(rhs_2);
            // `B⁰ᵀ` = `B⁰`ᵀ
            B0T = B0.transpose();
            // `B¹ᵀ` = `B¹`ᵀ
            B1T = B1.transpose();
        }
    };
    Neighborhoods _Neighborhoods;
    std::vector<int > VertexOneRing(int p0){
        return _Neighborhoods.VertexOneRing(p0);
    };
    std::vector<int > VertexOneRing(std::vector<int > p0){
        return _Neighborhoods.VertexOneRing(p0);
    };
    std::vector<int > Faces(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
        return _Neighborhoods.Faces(p0);
    };
    std::vector<int > Faces_0(int p0){
        return _Neighborhoods.Faces_0(p0);
    };
    std::vector<int > Faces_1(int p0){
        return _Neighborhoods.Faces_1(p0);
    };
    std::tuple< int, int > NeighborVerticesInFace(int p0,int p1){
        return _Neighborhoods.NeighborVerticesInFace(p0,p1);
    };
    heartlib(
        const FaceMesh & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    :
    _Neighborhoods(M)
    {
        // V, E, F = ElementSets( M )
        std::tuple< std::vector<int >, std::vector<int >, std::vector<int > > rhs = M.ElementSets();
        V = std::get<0>(rhs);
        E = std::get<1>(rhs);
        F = std::get<2>(rhs);
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        int dimf_0 = M.n_faces();
        const long dim_0 = x.size();
        this->M = M;
        this->x = x;
    
    }
};





