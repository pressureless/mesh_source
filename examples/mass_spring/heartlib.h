/*
ElementSets from MeshConnectivity
VertexOneRing from TetrahderonNeighborhoods(M)
M : CellMesh
x_i ∈ ℝ^3: original positions
m ∈ ℝ: mass
damping ∈ ℝ: damping
K ∈ ℝ: stiffness
`Δt` ∈ ℝ: step size
bottom ∈ ℝ: ground height
V, E, F, T = ElementSets( M )


e(i, j) = ||x_i - x_j|| where i,j ∈ V

ComputeInternalForces(i, v, p) = tuple(ṽ, f+(0.0, -98.0, 0.0))  where i ∈ V, v_i ∈ ℝ^3, p_i ∈ ℝ^3,
f = (∑_(j ∈ VertexOneRing(i)) (-K) (||d|| - e(i, j)) d̄ 
where d = p_i - p_j,
d̄ = d/||d||),
ṽ = v_i exp(-`Δt` damping) + `Δt` f


ApplyForces(i, v, f, p) = tuple(ṽ, x̃) where i ∈ V, v_i ∈ ℝ^3, f_i ∈ ℝ^3, p_i ∈ ℝ^3,
a = f_i / m,
v̄ = v_i + a `Δt`,
ṽ = { (0, -v̄_2, 0) if p_i,2 < bottom
      v̄ otherwise,
p̃ = { (p_i,1, bottom, p_i,3) if p_i,2 < bottom
     p_i otherwise,
x̃ = p̃ + ṽ `Δt`


*/
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include <algorithm>
#include "type_helper.h"
#include "CellMesh.h"

using namespace iheartmesh;
struct heartlib {
    using DT__ = double;
    using MatrixD__ = Eigen::MatrixXd;
    using VectorD__ = Eigen::VectorXd;
    std::vector<int > V;
    std::vector<int > E;
    std::vector<int > F;
    std::vector<int > T;
    CellMesh M;
    std::vector<Eigen::Matrix<double, 3, 1>> x;
    double K;
    double Δt;
    double damping;
    double m;
    double bottom;
    double e(
        const int & i,
        const int & j)
    {
        assert( std::binary_search(V.begin(), V.end(), i) );
        assert( std::binary_search(V.begin(), V.end(), j) );

        return (this->x.at(i) - this->x.at(j)).template lpNorm<2>();    
    }
    template<typename REAL>
    std::tuple< Eigen::Matrix<REAL, 3, 1>, Eigen::Matrix<REAL, 3, 1> > ComputeInternalForces(
        const int & i,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & v,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & p)
    {
        const long dim_1 = v.size();
        assert( p.size() == dim_1 );
        assert( std::binary_search(V.begin(), V.end(), i) );

        // f = (∑_(j ∈ VertexOneRing(i)) (-K) (||d|| - e(i, j)) d̄ 
        // where d = p_i - p_j,
        // d̄ = d/||d||)
        MatrixD__ sum_0 = MatrixD__::Zero(3, 1);
        for(int j : this->VertexOneRing(i)){
                // d = p_i - p_j
            Eigen::Matrix<REAL, 3, 1> d = p.at(i) - p.at(j);
                // d̄ = d/||d||
            Eigen::Matrix<REAL, 3, 1> d̄ = d / REAL((d).template lpNorm<2>());
            sum_0 += (-this->K) * ((d).template lpNorm<2>() - e(i, j)) * d̄;
        }
        Eigen::Matrix<REAL, 3, 1> f = (sum_0);

        // ṽ = v_i exp(-`Δt` damping) + `Δt` f
        Eigen::Matrix<REAL, 3, 1> ṽ = v.at(i) * exp(-this->Δt * this->damping) + this->Δt * f;
        Eigen::Matrix<REAL, 3, 1> ComputeInternalForces_0;
        ComputeInternalForces_0 << 0.0, -98.0, 0.0;
        return std::tuple<Eigen::Matrix<REAL, 3, 1>,Eigen::Matrix<REAL, 3, 1> >{ ṽ,f + ComputeInternalForces_0 };    
    }
    template<typename REAL>
    std::tuple< Eigen::Matrix<REAL, 3, 1>, Eigen::Matrix<REAL, 3, 1> > ApplyForces(
        const int & i,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & v,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & f,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & p)
    {
        const long dim_2 = v.size();
        assert( f.size() == dim_2 );
        assert( p.size() == dim_2 );
        assert( std::binary_search(V.begin(), V.end(), i) );

        // a = f_i / m
        Eigen::Matrix<REAL, 3, 1> a = f.at(i) / REAL(this->m);

        // v̄ = v_i + a `Δt`
        Eigen::Matrix<REAL, 3, 1> v̄ = v.at(i) + a * this->Δt;

        // ṽ = { (0, -v̄_2, 0) if p_i,2 < bottom
        //       v̄ otherwise
        Eigen::Matrix<REAL, 3, 1> ṽ;
        if(p.at(i)[2-1] < this->bottom){
            Eigen::Matrix<REAL, 3, 1> ṽ_0;
        ṽ_0 << 0, -v̄[2-1], 0;
            ṽ = ṽ_0;
        }
        else{
            ṽ = v̄;
        }

        // p̃ = { (p_i,1, bottom, p_i,3) if p_i,2 < bottom
        //      p_i otherwise
        Eigen::Matrix<REAL, 3, 1> p̃;
        if(p.at(i)[2-1] < this->bottom){
            Eigen::Matrix<REAL, 3, 1> p̃_0;
        p̃_0 << p.at(i)[1-1], this->bottom, p.at(i)[3-1];
            p̃ = p̃_0;
        }
        else{
            p̃ = p.at(i);
        }

        // x̃ = p̃ + ṽ `Δt`
        Eigen::Matrix<REAL, 3, 1> x̃ = p̃ + ṽ * this->Δt;
        return std::tuple<Eigen::Matrix<REAL, 3, 1>,Eigen::Matrix<REAL, 3, 1> >{ ṽ,x̃ };    
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
    std::vector<int > VertexOneRing(int p0){
        return _TetrahderonNeighborhoods.VertexOneRing(p0);
    };
    std::vector<int > VertexOneRing(std::vector<int > p0){
        return _TetrahderonNeighborhoods.VertexOneRing(p0);
    };
    heartlib(
        const CellMesh & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x,
        const double & m,
        const double & damping,
        const double & K,
        const double & Δt,
        const double & bottom)
    :
    _TetrahderonNeighborhoods(M)
    {
        // V, E, F, T = ElementSets( M )
        std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > rhs = M.ElementSets();
        V = std::get<0>(rhs);
        E = std::get<1>(rhs);
        F = std::get<2>(rhs);
        T = std::get<3>(rhs);
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        int dimf_0 = M.n_faces();
        int dimt_0 = M.n_tets();
        const long dim_0 = x.size();
        this->M = M;
        this->x = x;
        this->K = K;
        this->Δt = Δt;
        this->damping = damping;
        this->m = m;
        this->bottom = bottom;
    
    }
};





