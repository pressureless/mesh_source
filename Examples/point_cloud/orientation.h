/*
ElementSets from MeshConnectivity
VertexOneRing, Vertices from PointCloudNeighborhoods(M)
M : PointCloud 
s ∈ ℝ^r
n_i ∈ ℝ^o

V, E = ElementSets( M )

energy_i(s) = ∑_(j ∈ VertexOneRing(i)) ‖ s_i n_i - s_j n_j ‖² where i ∈ V, s ∈ ℝ^r


total = ∑_( v ∈ V ) energy_v(s) + 100 ∑_i (s_i² -1)² 
 
g = ∂total/∂s
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
#include "PointCloud.h"

using namespace iheartmesh;
struct orientation {
    using DT = autodiff::var;
    using MatrixD = Eigen::Matrix<autodiff::var, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorD = Eigen::Matrix<autodiff::var, Eigen::Dynamic, 1>;
    std::vector<int > V;
    std::vector<int > E;
    DT total;
    Eigen::VectorXd g;
    PointCloud M;
    VectorD s;
    std::vector<Eigen::VectorXd> n;
    template<typename REAL>
    REAL energy(
        const int & i,
        const Eigen::Matrix<REAL, Eigen::Dynamic, 1> & s)
    {
        const long r = s.size();
        assert( std::binary_search(V.begin(), V.end(), i) );

        REAL sum_0 = 0;
        for(int j : this->VertexOneRing(i)){
            sum_0 += pow((s[i] * this->n.at(i) - s[j] * this->n.at(j)).template lpNorm<2>(), 2);
        }
        return sum_0;    
    }
    using DT_ = double;
    using MatrixD_ = Eigen::MatrixXd;
    using VectorD_ = Eigen::VectorXd;
    struct PointCloudNeighborhoods {
        std::vector<int > V;
        std::vector<int > E;
        Eigen::SparseMatrix<int> dee0;
        Eigen::SparseMatrix<int> B0;
        PointCloud M;
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
            const std::vector<int >& rhs_1 = nonzeros(this->dee0.transpose() * M.vertices_to_vector(EdgeIndexset_1));
            intsect.reserve(std::min(lhs.size(), rhs_1.size()));
            std::set_intersection(lhs.begin(), lhs.end(), rhs_1.begin(), rhs_1.end(), std::back_inserter(intsect));
            std::vector<int > eset = intsect;
            return eset[1-1];    
        }
        using DT = double;
        using MatrixD = Eigen::MatrixXd;
        using VectorD = Eigen::VectorXd;
        struct FundamentalPointCloudAccessors {
            std::vector<int > V;
            std::vector<int > E;
            Eigen::SparseMatrix<int> dee0;
            Eigen::SparseMatrix<int> B0;
            PointCloud M;
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
            std::vector<int > Vertices(
                const int & e)
            {
                assert( std::binary_search(E.begin(), E.end(), e) );
                std::vector<int > Vertices_0set_0({e});
                if(Vertices_0set_0.size() > 1){
                    sort(Vertices_0set_0.begin(), Vertices_0set_0.end());
                    Vertices_0set_0.erase(unique(Vertices_0set_0.begin(), Vertices_0set_0.end() ), Vertices_0set_0.end());
                }
                return nonzeros(this->B0 * M.edges_to_vector(Vertices_0set_0));    
            }
            std::vector<int > Vertices(
                const std::vector<int > & H)
            {
                return nonzeros(this->B0 * M.edges_to_vector(H));    
            }
            std::vector<int > Edges(
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
            FundamentalPointCloudAccessors(const PointCloud & M)
            {
                // V, E = ElementSets( M )
                std::tuple< std::vector<int >, std::vector<int > > rhs = M.ElementSets();
                V = std::get<0>(rhs);
                E = std::get<1>(rhs);
                int dimv_0 = M.n_vertices();
                int dime_0 = M.n_edges();
                this->M = M;
                // dee0 = BoundaryMatrices(M)
                dee0 = M.BoundaryMatrices();
                // B0 = UnsignedBoundaryMatrices(M)
                B0 = M.UnsignedBoundaryMatrices();
            }
        };
        FundamentalPointCloudAccessors _FundamentalPointCloudAccessors;
        std::vector<int > Vertices(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
            return _FundamentalPointCloudAccessors.Vertices(p0);
        };
        std::vector<int > Vertices(int p0){
            return _FundamentalPointCloudAccessors.Vertices(p0);
        };
        std::vector<int > Vertices(std::vector<int > p0){
            return _FundamentalPointCloudAccessors.Vertices(p0);
        };
        std::vector<int > Edges(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
            return _FundamentalPointCloudAccessors.Edges(p0);
        };
        std::vector<int > Edges(int p0){
            return _FundamentalPointCloudAccessors.Edges(p0);
        };
        PointCloudNeighborhoods(const PointCloud & M)
        :
        _FundamentalPointCloudAccessors(M)
        {
            // V, E = ElementSets( M )
            std::tuple< std::vector<int >, std::vector<int > > rhs = M.ElementSets();
            V = std::get<0>(rhs);
            E = std::get<1>(rhs);
            int dimv_0 = M.n_vertices();
            int dime_0 = M.n_edges();
            this->M = M;
            // `∂0` = BoundaryMatrices(M)
            dee0 = M.BoundaryMatrices();
            // B0 = UnsignedBoundaryMatrices(M)
            B0 = M.UnsignedBoundaryMatrices();
        }
    };
    PointCloudNeighborhoods _PointCloudNeighborhoods;
    std::vector<int > VertexOneRing(int p0){
        return _PointCloudNeighborhoods.VertexOneRing(p0);
    };
    std::vector<int > VertexOneRing(std::vector<int > p0){
        return _PointCloudNeighborhoods.VertexOneRing(p0);
    };
    std::vector<int > Vertices(std::tuple< std::vector<int >, std::vector<int >, std::vector<int >, std::vector<int > > p0){
        return _PointCloudNeighborhoods.Vertices(p0);
    };
    std::vector<int > Vertices(int p0){
        return _PointCloudNeighborhoods.Vertices(p0);
    };
    std::vector<int > Vertices(std::vector<int > p0){
        return _PointCloudNeighborhoods.Vertices(p0);
    };
    orientation(
        const PointCloud & M,
        const Eigen::VectorXd & s,
        const std::vector<Eigen::VectorXd> & n)
    :
    _PointCloudNeighborhoods(M)
    {
        // V, E = ElementSets( M )
        std::tuple< std::vector<int >, std::vector<int > > rhs = M.ElementSets();
        V = std::get<0>(rhs);
        E = std::get<1>(rhs);
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        const long r = s.size();
        const long dim_0 = n.size();
        const long o = n[0].rows();
        assert( n.size() == dim_0 );
        for( const auto& el : n ) {
            assert( el.size() == o );
        }
        this->M = M;
        this->s = s;
        this->n = n;
        // total = ∑_( v ∈ V ) energy_v(s) + 100 ∑_i (s_i² -1)²
        DT sum_1 = 0;
        for(int v : this->V){
            sum_1 += energy(v, this->s);
        }
        DT sum_2 = 0;
        for(int i=1; i<=s.size(); i++){
            sum_2 += pow((pow(this->s[i-1], 2) - 1), 2);
        }
        total = sum_1 + 100 * sum_2;
        // g = ∂total/∂s
        g = gradient(total, this->s);
    }
};





