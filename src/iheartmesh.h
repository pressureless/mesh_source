/*
vec from linearalgebra
MeshSets, BoundaryMatrices, UnsignedBoundaryMatrices, NonZeros, IndicatorVector from MeshConnectivity
Vertices, Edges, Faces from FundamentalMeshAccessors(M)
M ∈ mesh 
V, E, F = MeshSets( M )
∂0, ∂1 = BoundaryMatrices(M) 
B0, B1 = UnsignedBoundaryMatrices(M)


VertexOneRing(v) = vertexset(NonZeros(B0 B0ᵀ IndicatorVector(M, {v}))) - {v} where v ∈ V
VertexOneRing(v) = vertexset(NonZeros(B0 B0ᵀ IndicatorVector(M, v))) - v where v ∈ {ℤ} vertices

EdgeIndex(i, j) = evec_1 where i,j ∈ V,
evec = vec(edgeset(NonZeros(∂0ᵀ IndicatorVector(M, {i}))) ∩ vertexset(NonZeros(∂0ᵀ IndicatorVector(M, {j}))))

NeighborVerticesInFace(f, v) = tuple(vvec1_1, vvec2_1) where v ∈ V, f ∈ F,
es = edgeset(NonZeros(∂1 IndicatorVector(M, {f}))),
nes = { s for s ∈ es if ∂0_v,s != 0 },
eset1 = { e for e ∈ nes if ∂1_e,f ∂0_v,e == -1},
vset1 = vertexset(NonZeros(B0 IndicatorVector(M, eset1))),
vvec1 = vec(vset1 - {v}),
eset2 = { e for e ∈ nes if ∂1_e,f ∂0_v,e == 1 },
vset2 = vertexset(NonZeros(B0 IndicatorVector(M, eset2))),
vvec2 = vec(vset2 - {v})

NextVerticeInFace(f, v) = vset_1 where v ∈ V, f ∈ F,
eset = { e for e ∈ Edges(f) if ∂1_e,f ∂0_v,e == -1},
vset = Vertices(eset) - {v}


OrientedVertices(f) = tuple(vvec_1, i, j)    where f ∈ F,
vs = Vertices(f),
vvec = vec(vs),
i,j = NeighborVerticesInFace(f, vvec_1)

Diamond(e) = SimplicialSet(Vertices(e), {e}, Faces(e)) where e ∈ E

OppositeVertices(e) = tuple(evec_1, evec_2) where e ∈ E,
evec = vec(Vertices(Faces(e)) \ Vertices(e))

FaceIndex(i,j,k) = fvec_1  where i,j,k ∈ V,
ufv = (B0 B1)ᵀ,
iface = faceset(NonZeros(ufv  IndicatorVector(M, {i}))),
jface = faceset(NonZeros(ufv  IndicatorVector(M, {j}))),
kface = faceset(NonZeros(ufv IndicatorVector(M, {k}))),
fvec = vec(iface ∩ jface ∩ kface)

OrientedVertices(i,j,k) = NeighborVerticesInFace(f, i)  where i,j,k ∈ V,
f = FaceIndex(i, j, k)

*/
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include <algorithm>
#include "TriangleMesh.h" 

struct iheartmesh {
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
        std::vector<int > stdv = intsect;
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
        std::vector<int > stdv_1 = difference_2;
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
        std::vector<int > stdv_2 = difference_3;
        Eigen::VectorXi vec_2(Eigen::Map<Eigen::VectorXi>(&stdv_2[0], stdv_2.size()));
        // vvec2 = vec(vset2 - {v})
        Eigen::VectorXi vvec2 = vec_2;
        return std::tuple<int,int >{ vvec1[1-1],vvec2[1-1] };    
    }
    int NextVerticeInFace(
        const int & f,
        const int & v)
    {
        assert( std::binary_search(F.begin(), F.end(), f) );

        std::vector<int > NextVerticeInFaceset_0;
        const std::vector<int >& range_3 = Edges_1(f);
        NextVerticeInFaceset_0.reserve(range_3.size());
        for(int e : range_3){
            if(dee1.coeff(e, f) * dee0.coeff(v, e) == -1){
                NextVerticeInFaceset_0.push_back(e);
            }
        }
        if(NextVerticeInFaceset_0.size() > 1){
            sort(NextVerticeInFaceset_0.begin(), NextVerticeInFaceset_0.end());
            NextVerticeInFaceset_0.erase(unique(NextVerticeInFaceset_0.begin(), NextVerticeInFaceset_0.end() ), NextVerticeInFaceset_0.end());
        }
        // eset = { e for e ∈ Edges(f) if ∂1_e,f ∂0_v,e == -1}
        std::vector<int > eset = NextVerticeInFaceset_0;

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
        // vset = Vertices(eset) - {v}
        std::vector<int > vset = difference_4;
        return vset[1-1];    
    }
    std::tuple< int, int, int > OrientedVertices(
        const int & f)
    {
        assert( std::binary_search(F.begin(), F.end(), f) );

        // vs = Vertices(f)
        std::vector<int > vs = Vertices_0(f);

        std::vector<int > stdv_3 = vs;
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

        std::vector<int > difference_5;
        const std::vector<int >& lhs_diff_5 = Vertices_1(Faces_1(e));
        const std::vector<int >& rhs_diff_5 = Vertices_2(e);
        difference_5.reserve(lhs_diff_5.size());
        std::set_difference(lhs_diff_5.begin(), lhs_diff_5.end(), rhs_diff_5.begin(), rhs_diff_5.end(), std::back_inserter(difference_5));
        std::vector<int > stdv_4 = difference_5;
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
        std::vector<int > stdv_5 = intsect_2;
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
    iheartmesh(const TriangleMesh & M)
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
 