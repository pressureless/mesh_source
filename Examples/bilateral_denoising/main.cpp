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
#include <igl/readOFF.h>
#include "MeshHelper.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

struct iheartla {

    std::vector<Eigen::Matrix<double, 3, 1>> x;
    Eigen::Matrix<double, 3, 1> GetVertexNormal(
        const int & i)
    {
        Eigen::MatrixXd sum_0 = Eigen::MatrixXd::Zero(3, 1);
        for(int f : FaceNeighbors(i)){
                // j, k = GetNeighborVerticesInFace(f, i)
            std::tuple< int, int > tuple = GetNeighborVerticesInFace(f, i);
            int j = std::get<0>(tuple);
            int k = std::get<1>(tuple);
                // n = (x_j- x_i)×(x_k-x_i)
            Eigen::Matrix<double, 3, 1> n = ((x.at(j) - x.at(i))).cross((x.at(k) - x.at(i)));
            sum_0 += n * (((x.at(j) - x.at(i))).cross((x.at(k) - x.at(i)))).lpNorm<2>();
        }
        // w = (sum_(f ∈ FaceNeighbors(i)) n ||(x_j- x_i)×(x_k-x_i)||
    // where n = (x_j- x_i)×(x_k-x_i),
    // j, k = GetNeighborVerticesInFace(f, i) )
        Eigen::Matrix<double, 3, 1> w = (sum_0);
        return w / double((w).lpNorm<2>());    
    }
    double CalSigmaC(
        const int & i)
    {
        std::set<double > CalSigmaCset_0;
        for(int v : VertexOneRing(i)){
            CalSigmaCset_0.insert((x.at(i) - x.at(v)).lpNorm<2>());
        }
        return *(CalSigmaCset_0).begin();    
    }
    double CalSigmaS(
        const int & i,
        const std::set<int > & N)
    {
        double CalSigmaS_ret;
        // n = GetVertexNormal(i)
        Eigen::Matrix<double, 3, 1> n = GetVertexNormal(i);

        double sum_3 = 0;
        for(int v : N){
                // t = sqrt(((x_v - x_i)⋅n)²)
            double t = sqrt(pow((((x.at(v) - x.at(i))).dot(n)), 2));
            sum_3 += t;
        }
        // t = sqrt(((x_v - x_i)⋅n)²)
        double avg = (sum_3) / double((N).size());

        double sum_4 = 0;
        for(int v : N){
                // t = sqrt(((x_v - x_i)⋅n)²)
            double t = sqrt(pow((((x.at(v) - x.at(i))).dot(n)), 2));
            sum_4 += pow((t - avg), 2);
        }
        // t = sqrt(((x_v - x_i)⋅n)²)
        double sqs = (sum_4);

        // offset = sqs / |N|
        double offset = sqs / double((N).size());
        if(sqrt(offset) < 1.0E-12){
            CalSigmaS_ret = sqrt(offset) + 1.0E-12;
        }
        else{
            CalSigmaS_ret = sqrt(offset);
        }
        return CalSigmaS_ret;    
    }
    std::set<int > GetAdaptiveVertexNeighbor(
        const int & i,
        const std::set<int > & n,
        const double & sigma)
    {
        std::set<int > GetAdaptiveVertexNeighbor_ret;
        std::set<int > GetAdaptiveVertexNeighborset_0;
        for(int v : VertexOneRing(n)){
            if((x.at(i) - x.at(v)).lpNorm<2>() < 2 * sigma){
                GetAdaptiveVertexNeighborset_0.insert(v);
            }
        }
        std::set<int > uni;
        std::set_union(GetAdaptiveVertexNeighborset_0.begin(), GetAdaptiveVertexNeighborset_0.end(), n.begin(), n.end(), std::inserter(uni, uni.begin()));
        // target = {v for v ∈ VertexOneRing(n) if ||x_i-x_v||< 2 sigma} + n
        std::set<int > target = uni;
        if((n).size() == (target).size()){
            GetAdaptiveVertexNeighbor_ret = n;
        }
        else{
            GetAdaptiveVertexNeighbor_ret = GetAdaptiveVertexNeighbor(i, target, sigma);
        }
        return GetAdaptiveVertexNeighbor_ret;    
    }
    Eigen::Matrix<double, 3, 1> DenoisePoint(
        const int & i)
    {
        // n = GetVertexNormal(i)
        Eigen::Matrix<double, 3, 1> n = GetVertexNormal(i);

        // `σc` = CalSigmaC(i)
        double σc = CalSigmaC(i);

        std::set<int > DenoisePointset_0({i});
        // neighbors = GetAdaptiveVertexNeighbor(i, {i}, `σc`)
        std::set<int > neighbors = GetAdaptiveVertexNeighbor(i, DenoisePointset_0, σc);

        // `σs` = CalSigmaS(i, neighbors)
        double σs = CalSigmaS(i, neighbors);

        double sum_6 = 0;
        for(int v : neighbors){
                // `ws` = exp(-h²/(2`σs`²))
            double t = (x.at(i) - x.at(v)).lpNorm<2>();
                // `ws` = exp(-h²/(2`σs`²))
            double h = (n).dot(x.at(v) - x.at(i));
                // `ws` = exp(-h²/(2`σs`²))
            double wc = exp(-pow(t, 2) / double((2 * pow(σc, 2))));
                // `ws` = exp(-h²/(2`σs`²))
            double ws = exp(-pow(h, 2) / double((2 * pow(σs, 2))));
            sum_6 += (wc * ws) * h;
        }
        // `ws` = exp(-h²/(2`σs`²))
        double s = (sum_6);

        double sum_7 = 0;
        for(int v : neighbors){
                // `ws` = exp(-h²/(2`σs`²))
            double t = (x.at(i) - x.at(v)).lpNorm<2>();
                // `ws` = exp(-h²/(2`σs`²))
            double h = (n).dot(x.at(v) - x.at(i));
                // `ws` = exp(-h²/(2`σs`²))
            double wc = exp(-pow(t, 2) / double((2 * pow(σc, 2))));
                // `ws` = exp(-h²/(2`σs`²))
            double ws = exp(-pow(h, 2) / double((2 * pow(σs, 2))));
            sum_7 += wc * ws;
        }
        // `ws` = exp(-h²/(2`σs`²))
        double norm = (sum_7);
        return x.at(i) + n * (s / double(norm));    
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
            std::set_intersection(nonzeros(ve.transpose() * M.vertices_to_vector(GetEdgeIndexset_0)).begin(), nonzeros(ve.transpose() * M.vertices_to_vector(GetEdgeIndexset_0)).end(), nonzeros(ve.transpose() * M.vertices_to_vector(GetEdgeIndexset_1)).begin(), nonzeros(ve.transpose() * M.vertices_to_vector(GetEdgeIndexset_1)).end(), std::inserter(intsect, intsect.begin()));
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
    std::set<int > FaceNeighbors(int p0){
        return _FundamentalMeshAccessors.FaceNeighbors(p0);
    };
    std::set<int > FaceNeighbors_0(int p0){
        return _FundamentalMeshAccessors.FaceNeighbors_0(p0);
    };
    std::set<int > VertexOneRing(int p0){
        return _FundamentalMeshAccessors.VertexOneRing(p0);
    };
    std::set<int > VertexOneRing(std::set<int > p0){
        return _FundamentalMeshAccessors.VertexOneRing(p0);
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

Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
TriangleMesh triangle_mesh;

std::vector<Eigen::Matrix<double, 3, 1>> P;

void update(){
    iheartla ihla(triangle_mesh, P); 
    std::cout<<"before"<<std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout<<"i: "<<i<<", "<<P[i]<<std::endl;
    } 
    std::vector<Eigen::Matrix<double, 3, 1>> NP;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Eigen::Matrix<double, 3, 1> new_pos = ihla.DenoisePoint(i);
        NP.push_back(new_pos);
    } 
    P = NP;
    // std::cout<<"after"<<std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout<<"i: "<<i<<", "<<P[i]<<std::endl;
    } 
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(P);
}

void myCallback()
{
    if (ImGui::Button("Run/Stop Simulation"))
    {
        std::cout<<"Run stop"<<std::endl;
    }
    // ImGui::SameLine();
    if (ImGui::Button("One step")){
        std::cout<<"one step"<<std::endl;
        update();
    } 
    if (ImGui::Button("Five steps")){
        for (int i = 0; i < 5; ++i)
        {
            update();
        }
    } 
}

int main(int argc, const char * argv[]) {
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/Mesh_Denoiseing_BilateralFilter/Noisy.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/fast-mesh-denoising/meshes/Noisy/block_n1.obj", meshV, meshF);
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/bumpy.off", meshV, meshF); // 69KB 5mins
    igl::readOBJ("/Users/pressure/Downloads/GuidedDenoising/models/compareWang/torusnoise.obj", meshV, meshF); // 177KB 20 mins
    

    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    triangle_mesh.initialize(meshV, meshF);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::state::userCallback = myCallback;

    for (int i = 0; i < meshV.rows(); ++i)
    {
        P.push_back(meshV.row(i).transpose());
    }
    // update();
    // iheartla ihla(triangle_mesh, P); 
    // // std::cout<<"before"<<std::endl;

    // std::cout<<"nn:\n"<<std::endl;
    // i, j, k = ihla.Vertices(0);
    // // print_set();
    // std::cout<<"i:"<<i<<", j:"<<j<<", k:"<<k<<std::endl;
    // std::cout<<"nn:\n"<<P[0]<<std::endl;
    // polyscope::getSurfaceMesh("my mesh")->addVertexVectorQuantity("VertexNormal", N); 
    polyscope::show();
    return 0;
}
