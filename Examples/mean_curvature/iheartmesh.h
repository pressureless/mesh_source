/*
ElementSets from MeshConnectivity
Faces, EdgeIndex, VertexOneRing, OppositeVertices, OrientedOppositeFaces, OrientedVertices, NeighborVerticesInFace from Neighborhoods(M)
M : TriangleMesh

t ∈ ℝ: step length
V, E, F = ElementSets( M )

clamp(v) = { -bound if v < -bound
             bound if v > bound
         v otherwise where v ∈ ℝ,
bound = 19.1

area(f, p, x) = { 0 if A=0
        ½A if dotp < 0
                  ¼A if dotq <0 or dotr <0
                  ⅛(cotq ||pr||² + cotr ||pq||² ) otherwise where f ∈ F, x_i ∈ ℝ^3, p ∈ V,
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
                    0 otherwise where i,j,k ∈ V, x_i ∈ ℝ^3 ,
oj, oi = OrientedVertices(k, j, i),
cos = (x_oj - x_k)⋅(x_oi-x_k),
sin = ||(x_oj - x_k)×(x_oi-x_k)||


Ax(i, x) = x_i - t w K  where i ∈ V, x_i ∈ ℝ^3,
A = (sum_(f ∈ Faces(i)) area(f, i, x)),
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
#include "type_helper.h"
#include "TriangleMesh.h"

using DT = double;
using MatrixD = Eigen::MatrixXd;
using VectorD = Eigen::VectorXd;
struct iheartmesh {
    std::vector<int > V;
    std::vector<int > E;
    std::vector<int > F;
    TriangleMesh M;
    double t;
    template<typename REAL>
    REAL clamp(
        const REAL & v)
    {
        REAL clamp_ret;
        // bound = 19.1
        REAL bound = 19.1;
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
    template<typename REAL>
    REAL area(
        const int & f,
        const int & p,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & x)
    {
        const long dim_0 = x.size();
        assert( std::binary_search(F.begin(), F.end(), f) );
        assert( std::binary_search(V.begin(), V.end(), p) );

        REAL area_ret;
        // q,r = NeighborVerticesInFace(f, p)
        std::tuple< int, int > rhs_1 = this->NeighborVerticesInFace(f, p);
        int q = std::get<0>(rhs_1);
        int r = std::get<1>(rhs_1);

        // pq = x_q - x_p
        Eigen::Matrix<REAL, 3, 1> pq = x.at(q) - x.at(p);

        // qr = x_r - x_q
        Eigen::Matrix<REAL, 3, 1> qr = x.at(r) - x.at(q);

        // pr = x_r - x_p
        Eigen::Matrix<REAL, 3, 1> pr = x.at(r) - x.at(p);

        // A = ½||pq×pr||
        REAL A = (1/REAL(2)) * ((pq).cross(pr)).template lpNorm<2>();

        // dotp = pq ⋅ pr
        REAL dotp = (pq).dot(pr);

        // dotq = (x_q-x_r) ⋅ pq
        REAL dotq = ((x.at(q) - x.at(r))).dot(pq);

        // dotr = qr⋅ pr
        REAL dotr = (qr).dot(pr);

        // cotq = clamp(dotq/(2A))
        REAL cotq = clamp(dotq / REAL((2 * A)));

        // cotr = clamp(dotr/(2A))
        REAL cotr = clamp(dotr / REAL((2 * A)));
        if(A == 0){
            area_ret = 0;
        }
        else if(dotp < 0){
            area_ret = (1/REAL(2)) * A;
        }
        else if((dotq < 0) || (dotr < 0)){
            area_ret = (1/REAL(4)) * A;
        }
        else{
            area_ret = (1/REAL(8)) * (cotq * pow((pr).template lpNorm<2>(), 2) + cotr * pow((pq).template lpNorm<2>(), 2));
        }
        return area_ret;    
    }
    template<typename REAL>
    REAL cot(
        const int & k,
        const int & j,
        const int & i,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & x)
    {
        const long dim_1 = x.size();
        assert( std::binary_search(V.begin(), V.end(), i) );
        assert( std::binary_search(V.begin(), V.end(), j) );
        assert( std::binary_search(V.begin(), V.end(), k) );

        REAL cot_ret;
        // oj, oi = OrientedVertices(k, j, i)
        std::tuple< int, int > rhs_2 = this->OrientedVertices(k, j, i);
        int oj = std::get<0>(rhs_2);
        int oi = std::get<1>(rhs_2);

        // cos = (x_oj - x_k)⋅(x_oi-x_k)
        REAL cos = ((x.at(oj) - x.at(k))).dot((x.at(oi) - x.at(k)));

        // sin = ||(x_oj - x_k)×(x_oi-x_k)||
        REAL sin = (((x.at(oj) - x.at(k))).cross((x.at(oi) - x.at(k)))).template lpNorm<2>();
        if(sin != 0){
            cot_ret = clamp(cos / REAL(sin));
        }
        else{
            cot_ret = 0;
        }
        return cot_ret;    
    }
    template<typename REAL>
    Eigen::Matrix<REAL, 3, 1> Ax(
        const int & i,
        const std::vector<Eigen::Matrix<REAL, 3, 1>> & x)
    {
        const long dim_2 = x.size();
        assert( std::binary_search(V.begin(), V.end(), i) );

        // A = (sum_(f ∈ Faces(i)) area(f, i, x))
        REAL sum_0 = 0;
        for(int f : Faces_0(i)){
            sum_0 += area(f, i, x);
        }
        REAL A = (sum_0);

        // w = { 1/(2A) if A≠0
        //       0      otherwise
        REAL w;
        if(A != 0){
            w = 1 / REAL((2 * A));
        }
        else{
            w = 0;
        }

        // K = (sum_(j ∈ VertexOneRing(i)) max(`cot(α)` + `cot(β)`,0)(x_j - x_i) 
        // where k, l = OppositeVertices(EdgeIndex(i,j)),
        // `cot(α)` = cot(k, j, i, x),
        // `cot(β)` = cot(l, i, j, x) )
        MatrixD sum_1 = MatrixD::Zero(3, 1);
        for(int j : this->VertexOneRing(i)){
                // k, l = OppositeVertices(EdgeIndex(i,j))
            std::tuple< int, int > rhs_3 = this->OppositeVertices(this->EdgeIndex(i, j));
            int k = std::get<0>(rhs_3);
            int l = std::get<1>(rhs_3);
                // `cot(α)` = cot(k, j, i, x)
            REAL cotα = cot(k, j, i, x);
                // `cot(β)` = cot(l, i, j, x)
            REAL cotβ = cot(l, i, j, x);
            sum_1 += std::max({REAL(cotα + cotβ), REAL(0)}) * (x.at(j) - x.at(i));
        }
        Eigen::Matrix<REAL, 3, 1> K = (sum_1);
        return x.at(i) - this->t * w * K;    
    }
    using DT_ = double;
    using MatrixD_ = Eigen::MatrixXd;
    using VectorD_ = Eigen::VectorXd;
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
            // V, E, F = ElementSets( M )4
            int e = EdgeIndex(i, j);
            // V, E, F = ElementSets( M )5
            std::vector<int > fset = Faces_1(e);
            // V, E, F = ElementSets( M )6
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
            // V, E, F = ElementSets( M )7
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
            // `∂0`, `∂1` = BoundaryMatrices(M)0
            std::vector<int > fset = Faces_1(e);
            // `∂0`, `∂1` = BoundaryMatrices(M)1
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
            // `∂0`, `∂1` = BoundaryMatrices(M)2
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
            // `∂0`, `∂1` = BoundaryMatrices(M)5
            std::tuple< int, int > rhs_5 = OppositeFaces(e);
            int firf = std::get<0>(rhs_5);
            int secf = std::get<1>(rhs_5);
            // `∂0`, `∂1` = BoundaryMatrices(M)6
            std::vector<int > difference_6;
            const std::vector<int >& lhs_diff_6 = Vertices_0(firf);
            const std::vector<int >& rhs_diff_6 = Vertices_2(e);
            difference_6.reserve(lhs_diff_6.size());
            std::set_difference(lhs_diff_6.begin(), lhs_diff_6.end(), rhs_diff_6.begin(), rhs_diff_6.end(), std::back_inserter(difference_6));
            std::vector<int > firv = difference_6;
            // `∂0`, `∂1` = BoundaryMatrices(M)7
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
            // B0, B1 = UnsignedBoundaryMatrices(M)0
            std::tuple< int, int > rhs_6 = OrientedOppositeFaces(i, j);
            int firf = std::get<0>(rhs_6);
            int secf = std::get<1>(rhs_6);
            // B0, B1 = UnsignedBoundaryMatrices(M)1
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
            // B0, B1 = UnsignedBoundaryMatrices(M)2
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
            // B0, B1 = UnsignedBoundaryMatrices(M)5
            Eigen::SparseMatrix<int> ufv = (this->B0 * this->B1).transpose();
            // B0, B1 = UnsignedBoundaryMatrices(M)6
            std::vector<int > FaceIndexset_0({i});
            if(FaceIndexset_0.size() > 1){
                sort(FaceIndexset_0.begin(), FaceIndexset_0.end());
                FaceIndexset_0.erase(unique(FaceIndexset_0.begin(), FaceIndexset_0.end() ), FaceIndexset_0.end());
            }
            std::vector<int > iface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_0));
            // B0, B1 = UnsignedBoundaryMatrices(M)7
            std::vector<int > FaceIndexset_1({j});
            if(FaceIndexset_1.size() > 1){
                sort(FaceIndexset_1.begin(), FaceIndexset_1.end());
                FaceIndexset_1.erase(unique(FaceIndexset_1.begin(), FaceIndexset_1.end() ), FaceIndexset_1.end());
            }
            std::vector<int > jface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_1));
            // B0, B1 = UnsignedBoundaryMatrices(M)8
            std::vector<int > FaceIndexset_2({k});
            if(FaceIndexset_2.size() > 1){
                sort(FaceIndexset_2.begin(), FaceIndexset_2.end());
                FaceIndexset_2.erase(unique(FaceIndexset_2.begin(), FaceIndexset_2.end() ), FaceIndexset_2.end());
            }
            std::vector<int > kface = nonzeros(ufv * M.vertices_to_vector(FaceIndexset_2));
            // B0, B1 = UnsignedBoundaryMatrices(M)9
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
            // f = FaceIndex(i, j, k)
            int f = FaceIndex(i, j, k);
            return NeighborVerticesInFace(f, i);    
        }
        using DT = double;
        using MatrixD = Eigen::MatrixXd;
        using VectorD = Eigen::VectorXd;
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
                return nonzeros(this->B0 * this->B1 * M.faces_to_vector(Vertices_0set_0));    
            }
            std::vector<int > Vertices_1(
                const std::vector<int > & G)
            {
                return nonzeros(this->B0 * this->B1 * M.faces_to_vector(G));    
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
                return nonzeros(this->B0 * M.edges_to_vector(Vertices_2set_0));    
            }
            std::vector<int > Vertices_3(
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
            FundamentalMeshAccessors(const TriangleMesh & M)
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
            // V, E, F = ElementSets( M )
            std::tuple< std::vector<int >, std::vector<int >, std::vector<int > > rhs = M.ElementSets();
            V = std::get<0>(rhs);
            E = std::get<1>(rhs);
            F = std::get<2>(rhs);
            int dimv_0 = M.n_vertices();
            int dime_0 = M.n_edges();
            int dimf_0 = M.n_faces();
            this->M = M;
            // `∂0`, `∂1` = BoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_1 = M.BoundaryMatrices();
            dee0 = std::get<0>(rhs_1);
            dee1 = std::get<1>(rhs_1);
            // B0, B1 = UnsignedBoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > rhs_2 = M.UnsignedBoundaryMatrices();
            B0 = std::get<0>(rhs_2);
            B1 = std::get<1>(rhs_2);
        }
    };
    Neighborhoods _Neighborhoods;
    std::vector<int > Faces(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
        return _Neighborhoods.Faces(p0);
    };
    std::vector<int > Faces_0(int p0){
        return _Neighborhoods.Faces_0(p0);
    };
    std::vector<int > Faces_1(int p0){
        return _Neighborhoods.Faces_1(p0);
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
    std::tuple< int, int > OppositeVertices(int p0,int p1){
        return _Neighborhoods.OppositeVertices(p0,p1);
    };
    std::tuple< int, int > OrientedOppositeFaces(int p0,int p1){
        return _Neighborhoods.OrientedOppositeFaces(p0,p1);
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
    iheartmesh(
        const TriangleMesh & M,
        const double & t)
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
        this->M = M;
        this->t = t;
    
    }
};
 