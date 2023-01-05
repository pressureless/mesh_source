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
    std::set<int > GetNextLevel(
        const std::vector<std::set<int >> & U)
    {
        const long dim_2 = U.size();
        std::set<int > union_0;
        for(int i=1; i<=U.size(); i++){
            std::set_union(union_0.begin(), union_0.end(), U.at(i-1).begin(), U.at(i-1).end(), std::inserter(union_0, union_0.begin()));
        }
        // s = ∪_i U_i
        std::set<int > s = union_0;

        // v = VertexOneRing(s)
        std::set<int > v = VertexOneRing(s);
        std::set<int > difference;
        std::set<int > lhs_diff = v;
        std::set<int > rhs_diff = s;
        std::set_difference(lhs_diff.begin(), lhs_diff.end(), rhs_diff.begin(), rhs_diff.end(), std::inserter(difference, difference.begin()));
        return difference;    
    }
    std::set<int > GetRangeLevel(
        const std::vector<std::set<int >> & U,
        const int & a,
        const int & b)
    {
        const long dim_3 = U.size();
        std::set<int > union_1;
        for(int i=a; i<=b; i++){
            std::set_union(union_1.begin(), union_1.end(), U.at(i).begin(), U.at(i).end(), std::inserter(union_1, union_1.begin()));
        }
        return union_1;    
    }
    struct FundamentalMeshAccessors {
        std::set<int > V;
        std::set<int > E;
        std::set<int > F;
        Eigen::SparseMatrix<int> ve;
        Eigen::SparseMatrix<int> ef;
        Eigen::SparseMatrix<int> uve;
        Eigen::SparseMatrix<int> uef;
        TriangleMesh M;
        std::set<int > Vertices(
            const std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > & S)
        {
            return std::get<1-1>(S);    
        }
        std::set<int > Vertices_0(
            const int & f)
        {
            assert( F.find(f) != F.end() );
            std::set<int > Vertices_0set_0({f});
            return nonzeros(uve * uef * M.faces_to_vector(Vertices_0set_0));    
        }
        std::set<int > Vertices_1(
            const std::set<int > & f)
        {
            return nonzeros(uve * uef * M.faces_to_vector(f));    
        }
        std::set<int > Vertices_2(
            const int & e)
        {
            assert( E.find(e) != E.end() );
            std::set<int > Vertices_2set_0({e});
            return nonzeros(uve * M.edges_to_vector(Vertices_2set_0));    
        }
        std::set<int > Vertices_3(
            const std::set<int > & e)
        {
            return nonzeros(uve * M.edges_to_vector(e));    
        }
        std::set<int > VertexOneRing(
            const int & v)
        {
            assert( V.find(v) != V.end() );
            std::set<int > VertexOneRingset_0({v});
            std::set<int > VertexOneRingset_1({v});
            std::set<int > difference;
            std::set<int > lhs_diff = nonzeros(uve * uve.transpose() * M.vertices_to_vector(VertexOneRingset_0));
            std::set<int > rhs_diff = VertexOneRingset_1;
            std::set_difference(lhs_diff.begin(), lhs_diff.end(), rhs_diff.begin(), rhs_diff.end(), std::inserter(difference, difference.begin()));
            return difference;    
        }
        std::set<int > VertexOneRing(
            const std::set<int > & v)
        {
            std::set<int > difference_1;
            std::set<int > lhs_diff_1 = nonzeros(uve * uve.transpose() * M.vertices_to_vector(v));
            std::set<int > rhs_diff_1 = v;
            std::set_difference(lhs_diff_1.begin(), lhs_diff_1.end(), rhs_diff_1.begin(), rhs_diff_1.end(), std::inserter(difference_1, difference_1.begin()));
            return difference_1;    
        }
        std::set<int > FaceNeighbors(
            const int & v)
        {
            assert( V.find(v) != V.end() );
            std::set<int > FaceNeighborsset_0({v});
            return nonzeros((uve * uef).transpose() * M.vertices_to_vector(FaceNeighborsset_0));    
        }
        std::set<int > FaceNeighbors_0(
            const int & e)
        {
            assert( E.find(e) != E.end() );
            std::set<int > FaceNeighbors_0set_0({e});
            return nonzeros(uef.transpose() * M.edges_to_vector(FaceNeighbors_0set_0));    
        }
        std::set<int > Faces(
            const std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > & S)
        {
            return std::get<3-1>(S);    
        }
        std::set<int > Edges(
            const std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > & S)
        {
            return std::get<2-1>(S);    
        }
        std::set<int > Edges(
            const int & f)
        {
            assert( F.find(f) != F.end() );
            std::set<int > Edges_0set_0({f});
            return nonzeros(uef * M.faces_to_vector(Edges_0set_0));    
        }
        int GetEdgeIndex(
            const int & i,
            const int & j)
        {
            assert( V.find(j) != V.end() );
            std::set<int > GetEdgeIndexset_0({i});
            std::set<int > GetEdgeIndexset_1({j});
            std::set<int > intsect;
            std::set<int > lhs = nonzeros(ve.transpose() * M.vertices_to_vector(GetEdgeIndexset_0));
            std::set<int > rhs = nonzeros(ve.transpose() * M.vertices_to_vector(GetEdgeIndexset_1));
            std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::inserter(intsect, intsect.begin()));
            std::vector<int> stdv(intsect.begin(), intsect.end());
            Eigen::VectorXi vec(Eigen::Map<Eigen::VectorXi>(&stdv[0], stdv.size()));
            // evec = vec(edgeset(NonZeros(veᵀ IndicatorVector(M, {i}))) ∩ vertexset(NonZeros(veᵀ IndicatorVector(M, {j}))))
            Eigen::VectorXi evec = vec;
            return evec[1-1];    
        }
        std::tuple< int, int > GetNeighborVerticesInFace(
            const int & f,
            const int & v)
        {
            assert( F.find(f) != F.end() );
            std::set<int > GetNeighborVerticesInFaceset_0({f});
            // es = edgeset(NonZeros(ef IndicatorVector(M, {f})))
            std::set<int > es = nonzeros(ef * M.faces_to_vector(GetNeighborVerticesInFaceset_0));
            std::set<int > GetNeighborVerticesInFaceset_1;
            for(int s : es){
                if(ve.coeff(v, s) != 0){
                    GetNeighborVerticesInFaceset_1.insert(s);
                }
            }
            // nes = { s for s ∈ es if ve_v,s != 0 }
            std::set<int > nes = GetNeighborVerticesInFaceset_1;
            std::set<int > GetNeighborVerticesInFaceset_2;
            for(int e : nes){
                if(ef.coeff(e, f) * ve.coeff(v, e) == -1){
                    GetNeighborVerticesInFaceset_2.insert(e);
                }
            }
            // eset1 = { e for e ∈ nes if ef_e,f ve_v,e == -1}
            std::set<int > eset1 = GetNeighborVerticesInFaceset_2;
            // vset1 = vertexset(NonZeros(uve IndicatorVector(M, eset1)))
            std::set<int > vset1 = nonzeros(uve * M.edges_to_vector(eset1));
            std::set<int > GetNeighborVerticesInFaceset_3({v});
            std::set<int > difference_2;
            std::set<int > lhs_diff_2 = vset1;
            std::set<int > rhs_diff_2 = GetNeighborVerticesInFaceset_3;
            std::set_difference(lhs_diff_2.begin(), lhs_diff_2.end(), rhs_diff_2.begin(), rhs_diff_2.end(), std::inserter(difference_2, difference_2.begin()));
            std::vector<int> stdv_1(difference_2.begin(), difference_2.end());
            Eigen::VectorXi vec_1(Eigen::Map<Eigen::VectorXi>(&stdv_1[0], stdv_1.size()));
            // vvec1 = vec(vset1 - {v})
            Eigen::VectorXi vvec1 = vec_1;
            std::set<int > GetNeighborVerticesInFaceset_4;
            for(int e : nes){
                if(ef.coeff(e, f) * ve.coeff(v, e) == 1){
                    GetNeighborVerticesInFaceset_4.insert(e);
                }
            }
            // eset2 = { e for e ∈ nes if ef_e,f ve_v,e == 1 }
            std::set<int > eset2 = GetNeighborVerticesInFaceset_4;
            // vset2 = vertexset(NonZeros(uve IndicatorVector(M, eset2)))
            std::set<int > vset2 = nonzeros(uve * M.edges_to_vector(eset2));
            std::set<int > GetNeighborVerticesInFaceset_5({v});
            std::set<int > difference_3;
            std::set<int > lhs_diff_3 = vset2;
            std::set<int > rhs_diff_3 = GetNeighborVerticesInFaceset_5;
            std::set_difference(lhs_diff_3.begin(), lhs_diff_3.end(), rhs_diff_3.begin(), rhs_diff_3.end(), std::inserter(difference_3, difference_3.begin()));
            std::vector<int> stdv_2(difference_3.begin(), difference_3.end());
            Eigen::VectorXi vec_2(Eigen::Map<Eigen::VectorXi>(&stdv_2[0], stdv_2.size()));
            // vvec2 = vec(vset2 - {v})
            Eigen::VectorXi vvec2 = vec_2;
            return std::tuple<int,int >{ vvec1[1-1],vvec2[1-1] };    
        }
        std::tuple< int, int, int > GetOrientedVertices(
            const int & f)
        {
            assert( F.find(f) != F.end() );
            // vs = Vertices(f)
            std::set<int > vs = Vertices_0(f);
            std::vector<int> stdv_3(vs.begin(), vs.end());
            Eigen::VectorXi vec_3(Eigen::Map<Eigen::VectorXi>(&stdv_3[0], stdv_3.size()));
            // vvec = vec(vs)
            Eigen::VectorXi vvec = vec_3;
            // i,j = GetNeighborVerticesInFace(f, vvec_1)
            std::tuple< int, int > tuple_3 = GetNeighborVerticesInFace(f, vvec[1-1]);
            int i = std::get<0>(tuple_3);
            int j = std::get<1>(tuple_3);
            return std::tuple<int,int,int >{ vvec[1-1],i,j };    
        }
        std::tuple< std::set<int >, std::set<int >, std::set<int >, std::set<int > > Diamond(
            const int & e)
        {
            assert( E.find(e) != E.end() );
            std::set<int > Diamondset_0({e});
        std::set<int > tetset;
            return std::tuple<std::set<int >,std::set<int >,std::set<int >,std::set<int > >{ Vertices_2(e),Diamondset_0,FaceNeighbors_0(e),tetset };    
        }
        std::tuple< int, int > OppositeVertices(
            const int & e)
        {
            assert( E.find(e) != E.end() );
            std::vector<int> stdv_4(Vertices(Diamond(e)).begin(), Vertices(Diamond(e)).end());
            Eigen::VectorXi vec_4(Eigen::Map<Eigen::VectorXi>(&stdv_4[0], stdv_4.size()));
            // evec = vec(Vertices(Diamond(e)))
            Eigen::VectorXi evec = vec_4;
            return std::tuple<int,int >{ evec[1-1],evec[2-1] };    
        }
        FundamentalMeshAccessors(const TriangleMesh & M)
        {
            // V, E, F = MeshSets( M )
            std::tuple< std::set<int >, std::set<int >, std::set<int > > tuple = M.MeshSets();
            V = std::get<0>(tuple);
            E = std::get<1>(tuple);
            F = std::get<2>(tuple);
            int dimv_0 = M.n_vertices();
            int dime_0 = M.n_edges();
            int dimf_0 = M.n_faces();
            this->M = M;
            // ve, ef = BoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_1 = M.BoundaryMatrices();
            ve = std::get<0>(tuple_1);
            ef = std::get<1>(tuple_1);
            // uve, uef = UnsignedBoundaryMatrices(M)
            std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > tuple_2 = M.UnsignedBoundaryMatrices();
            uve = std::get<0>(tuple_2);
            uef = std::get<1>(tuple_2);
        }
    };
    FundamentalMeshAccessors _FundamentalMeshAccessors;
    std::tuple< int, int > GetNeighborVerticesInFace(int p0,int p1){
        return _FundamentalMeshAccessors.GetNeighborVerticesInFace(p0,p1);
    };
    std::set<int > VertexOneRing(int p0){
        return _FundamentalMeshAccessors.VertexOneRing(p0);
    };
    std::set<int > VertexOneRing(std::set<int > p0){
        return _FundamentalMeshAccessors.VertexOneRing(p0);
    };
    std::set<int > FaceNeighbors(int p0){
        return _FundamentalMeshAccessors.FaceNeighbors(p0);
    };
    std::set<int > FaceNeighbors_0(int p0){
        return _FundamentalMeshAccessors.FaceNeighbors_0(p0);
    };
    iheartla(
        const TriangleMesh & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    :
    _FundamentalMeshAccessors(M)
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
    triangle_mesh.initialize(meshV, meshF);
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
    std::vector<std::set<int> > U;
    std::set<int> origin;
    origin.insert(cur);
    distance[cur] = 0;

    std::set<int> next = origin;
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
        std::set<int> v_set = ihla.GetRangeLevel(U, i, j);
        // std::cout<<"current v_set: "<<std::endl;
        // print_set(v_set);
        for (int v: v_set)
        {
            std::set<int> f_set = ihla.FaceNeighbors(v);
            for (int f: f_set)
            {
                std::tuple< int, int > v_tuple = ihla.GetNeighborVerticesInFace(f, v);
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
