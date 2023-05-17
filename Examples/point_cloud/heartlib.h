/*
svd from linearalgebra
ElementSets from MeshConnectivity
VertexOneRing from PointCloudNeighborhoods(M)
M : PointCloud 
x_i ∈ ℝ^3 

V, E = ElementSets( M )

Normal(v) = vv_*,3 where v ∈ V,
N = VertexOneRing(v),
p̄ = (∑_(n ∈ N) x_n)/|N|,
d = {x_v-p̄ for v ∈ N},
m_i,* = d_i,
u, `∑`, vv = svd(m)

 
cov(v) = ∑_(n ∈ N) (x_v-p̄)(x_v-p̄)ᵀ where v ∈ V,
N = VertexOneRing(v),
p̄ = (∑_(n ∈ N) x_n)/|N|


*/
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include <algorithm>
#include "type_helper.h"
#include "PointCloud.h"

using namespace iheartmesh;

using DT = double;
using MatrixD = Eigen::MatrixXd;
using VectorD = Eigen::VectorXd;
struct heartlib {
    std::vector<int > V;
    std::vector<int > E;
    PointCloud M;
    std::vector<Eigen::Matrix<double, 3, 1>> x;
    Eigen::Matrix<double, 3, 1> Normal(
        const int & v)
    {
        assert( std::binary_search(V.begin(), V.end(), v) );

        // N = VertexOneRing(v)
        std::vector<int > N = this->VertexOneRing(v);

        // p̄ = (∑_(n ∈ N) x_n)/|N|
        MatrixD sum_0 = MatrixD::Zero(3, 1);
        for(int n : N){
            sum_0 += this->x.at(n);
        }
        Eigen::Matrix<double, 3, 1> p̄ = (sum_0) / double(double((N).size()));

        // d = {x_v-p̄ for v ∈ N}
        std::vector<Eigen::Matrix<double, 3, 1> > Normalset_0;
        const std::vector<int >& range = N;
        Normalset_0.reserve(range.size());
        for(int v : range){
            Normalset_0.push_back(this->x.at(v) - p̄);
        }
        if(Normalset_0.size() > 1){
            sort(Normalset_0.begin(), Normalset_0.end(), [](const Eigen::Matrix<double, 3, 1> &lhs_, const Eigen::Matrix<double, 3, 1> &rhs_)
            {
                for (int si=0; si<lhs_.rows(); si++) {
                    if (lhs_(si) == rhs_(si)) {
                        continue;
                    }
                    else if (lhs_(si) > rhs_(si)) {
                        return false;
                    }
                    return true;
                }
                return false;
            });
            Normalset_0.erase(unique(Normalset_0.begin(), Normalset_0.end() ), Normalset_0.end());
        }
        std::vector<Eigen::Matrix<double, 3, 1> > d = Normalset_0;

        // m_i,* = d_i
        Eigen::MatrixXd m = MatrixD::Zero(d.size(), 3);
        for( int i=1; i<=d.size(); i++){
            m.row(i-1) = d[i-1];
        }

        Eigen::BDCSVD<Eigen::MatrixXd> svd(to_double(m), Eigen::ComputeFullU | Eigen::ComputeFullV);
        // u, `∑`, vv = svd(m)
        std::tuple< Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix<double, 3, 3> > rhs_1 = std::tuple< Eigen::MatrixXd, Eigen::VectorXd, Eigen::Matrix<double, 3, 3> >(svd.matrixU(), svd.singularValues(), svd.matrixV());
        Eigen::MatrixXd u = std::get<0>(rhs_1);
        Eigen::VectorXd n_ary_summation = std::get<1>(rhs_1);
        Eigen::Matrix<double, 3, 3> vv = std::get<2>(rhs_1);
        return vv.col(3-1);    
    }
    Eigen::Matrix<double, 3, 3> cov(
        const int & v)
    {
        assert( std::binary_search(V.begin(), V.end(), v) );

        // N = VertexOneRing(v)
        std::vector<int > N = this->VertexOneRing(v);

        // p̄ = (∑_(n ∈ N) x_n)/|N|
        MatrixD sum_1 = MatrixD::Zero(3, 1);
        for(int n : N){
            sum_1 += this->x.at(n);
        }
        Eigen::Matrix<double, 3, 1> p̄ = (sum_1) / double(double((N).size()));
        MatrixD sum_2 = MatrixD::Zero(3, 3);
        for(int n : N){
            sum_2 += (this->x.at(v) - p̄) * (this->x.at(v) - p̄).transpose();
        }
        return sum_2;    
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
    heartlib(
        const PointCloud & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    :
    _PointCloudNeighborhoods(M)
    {
        // V, E = ElementSets( M )
        std::tuple< std::vector<int >, std::vector<int > > rhs = M.ElementSets();
        V = std::get<0>(rhs);
        E = std::get<1>(rhs);
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        const long dim_0 = x.size();
        this->M = M;
        this->x = x;
    
    }
};
 