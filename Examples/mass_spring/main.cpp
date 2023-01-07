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
#include <igl/writeOBJ.h>
#include <igl/readOFF.h>
#include "MeshHelper.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

struct iheartla {

    std::vector<Eigen::Matrix<double, 3, 1>> x;
    double K;
    double dt;
    double damping;
    double m;
    double e(
        const int & i,
        const int & j)
    {
        return (x.at(i) - x.at(j)).lpNorm<2>();    
    }
    std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > computeInternalForces(
        const int & i,
        const std::vector<Eigen::Matrix<double, 3, 1>> & v,
        const std::vector<Eigen::Matrix<double, 3, 1>> & xn)
    {
        const long dim_1 = v.size();
        assert( xn.size() == dim_1 );

        Eigen::MatrixXd sum_0 = Eigen::MatrixXd::Zero(3, 1);
        for(int j : VertexOneRing(i)){
                // disp = xn_i - xn_j
            Eigen::Matrix<double, 3, 1> disp = xn.at(i) - xn.at(j);
                // dir = disp/||disp||
            Eigen::Matrix<double, 3, 1> dir = disp / double((disp).lpNorm<2>());
            sum_0 += (-K) * ((disp).lpNorm<2>() - e(i, j)) * dir;
        }
        // f = (sum_(j ∈ VertexOneRing(i)) (-K) (||disp|| - e(i, j)) dir
    // where disp = xn_i - xn_j,
    // dir = disp/||disp||)
        Eigen::Matrix<double, 3, 1> f = (sum_0);

        // vn = v_i exp(-dt damping) + dt f
        Eigen::Matrix<double, 3, 1> vn = v.at(i) * exp(-dt * damping) + dt * f;
        Eigen::Matrix<double, 3, 1> computeInternalForces_0;
        computeInternalForces_0 << 0.0, -98.0, 0.0;
        return std::tuple<Eigen::Matrix<double, 3, 1>,Eigen::Matrix<double, 3, 1> >{ vn,f + computeInternalForces_0 };    
    }
    std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > applyForces(
        const int & i,
        const std::vector<Eigen::Matrix<double, 3, 1>> & v,
        const std::vector<Eigen::Matrix<double, 3, 1>> & f,
        const std::vector<Eigen::Matrix<double, 3, 1>> & xn)
    {
        const long dim_2 = v.size();
        assert( f.size() == dim_2 );
        assert( xn.size() == dim_2 );

        // a = f_i / m
        Eigen::Matrix<double, 3, 1> a = f.at(i) / double(m);

        // vn = v_i + a dt
        Eigen::Matrix<double, 3, 1> vn = v.at(i) + a * dt;

        // xnn = xn_i + vn dt
        Eigen::Matrix<double, 3, 1> xnn = xn.at(i) + vn * dt;
        return std::tuple<Eigen::Matrix<double, 3, 1>,Eigen::Matrix<double, 3, 1> >{ vn,xnn };    
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
            std::set<int > op = intsect;
            std::vector<int> stdv(op.begin(), op.end());
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
            std::set<int > op_1 = difference_2;
            std::vector<int> stdv_1(op_1.begin(), op_1.end());
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
            std::set<int > op_2 = difference_3;
            std::vector<int> stdv_2(op_2.begin(), op_2.end());
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
            std::set<int > op_3 = vs;
            std::vector<int> stdv_3(op_3.begin(), op_3.end());
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
            std::set<int > difference_4;
            std::set<int > lhs_diff_4 = Vertices_1(FaceNeighbors_0(e));
            std::set<int > rhs_diff_4 = Vertices_2(e);
            std::set_difference(lhs_diff_4.begin(), lhs_diff_4.end(), rhs_diff_4.begin(), rhs_diff_4.end(), std::inserter(difference_4, difference_4.begin()));
            std::set<int > op_4 = difference_4;
            std::vector<int> stdv_4(op_4.begin(), op_4.end());
            Eigen::VectorXi vec_4(Eigen::Map<Eigen::VectorXi>(&stdv_4[0], stdv_4.size()));
            // evec = vec(Vertices(FaceNeighbors(e)) \ Vertices(e))
            Eigen::VectorXi evec = vec_4;
            return std::tuple<int,int >{ evec[1-1],evec[2-1] };    
        }
        int GetFaceIndex(
            const int & i,
            const int & j,
            const int & k)
        {
            assert( V.find(k) != V.end() );
            // ufv = (uve uef)ᵀ
            Eigen::SparseMatrix<int> ufv = (uve * uef).transpose();
            std::set<int > GetFaceIndexset_0({i});
            // iface = faceset(NonZeros(ufv  IndicatorVector(M, {i})))
            std::set<int > iface = nonzeros(ufv * M.vertices_to_vector(GetFaceIndexset_0));
            std::set<int > GetFaceIndexset_1({j});
            // jface = faceset(NonZeros(ufv  IndicatorVector(M, {j})))
            std::set<int > jface = nonzeros(ufv * M.vertices_to_vector(GetFaceIndexset_1));
            std::set<int > GetFaceIndexset_2({k});
            // kface = faceset(NonZeros(ufv IndicatorVector(M, {k})))
            std::set<int > kface = nonzeros(ufv * M.vertices_to_vector(GetFaceIndexset_2));
            std::set<int > intsect_1;
            std::set<int > lhs_1 = jface;
            std::set<int > rhs_1 = kface;
            std::set_intersection(lhs_1.begin(), lhs_1.end(), rhs_1.begin(), rhs_1.end(), std::inserter(intsect_1, intsect_1.begin()));
            std::set<int > intsect_2;
            std::set<int > lhs_2 = iface;
            std::set<int > rhs_2 = intsect_1;
            std::set_intersection(lhs_2.begin(), lhs_2.end(), rhs_2.begin(), rhs_2.end(), std::inserter(intsect_2, intsect_2.begin()));
            std::set<int > op_5 = intsect_2;
            std::vector<int> stdv_5(op_5.begin(), op_5.end());
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
            assert( V.find(k) != V.end() );
            // f = GetFaceIndex(i, j, k)
            int f = GetFaceIndex(i, j, k);
            return GetNeighborVerticesInFace(f, i);    
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
    std::set<int > FaceNeighbors(int p0){
        return _FundamentalMeshAccessors.FaceNeighbors(p0);
    };
    std::set<int > FaceNeighbors_0(int p0){
        return _FundamentalMeshAccessors.FaceNeighbors_0(p0);
    };
    int GetEdgeIndex(int p0,int p1){
        return _FundamentalMeshAccessors.GetEdgeIndex(p0,p1);
    };
    std::set<int > VertexOneRing(int p0){
        return _FundamentalMeshAccessors.VertexOneRing(p0);
    };
    std::set<int > VertexOneRing(std::set<int > p0){
        return _FundamentalMeshAccessors.VertexOneRing(p0);
    };
    std::tuple< int, int > OppositeVertices(int p0){
        return _FundamentalMeshAccessors.OppositeVertices(p0);
    };
    std::tuple< int, int, int > GetOrientedVertices(int p0){
        return _FundamentalMeshAccessors.GetOrientedVertices(p0);
    };
    std::tuple< int, int > OrientedVertices(int p0,int p1,int p2){
        return _FundamentalMeshAccessors.OrientedVertices(p0,p1,p2);
    };
    std::tuple< int, int > GetNeighborVerticesInFace(int p0,int p1){
        return _FundamentalMeshAccessors.GetNeighborVerticesInFace(p0,p1);
    };
    iheartla(
        const TriangleMesh & M,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x,
        const double & m,
        const double & damping,
        const double & K,
        const double & dt)
    :
    _FundamentalMeshAccessors(M)
    {
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        int dimf_0 = M.n_faces();
        const long dim_0 = x.size();
        this->x = x;
        this->K = K;
        this->dt = dt;
        this->damping = damping;
        this->m = m;
    
    }
};

Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
TriangleMesh triangle_mesh;

double mass = 1.0;
double stiffness = 5e5;
double damping = 5;
double bottom_z = -70.0;
double dt = 2e-4;
double eps = 1e-6;

std::vector<Eigen::Matrix<double, 3, 1>> OriginalPosition;
std::vector<Eigen::Matrix<double, 3, 1>> Position;
std::vector<Eigen::Matrix<double, 3, 1>> Velocity;
std::vector<Eigen::Matrix<double, 3, 1>> Force;
 
void update(){
    iheartla ihla(triangle_mesh, OriginalPosition, mass, damping, stiffness, dt);
    for (int i = 0; i < meshV.rows(); ++i)
    {
        //
        Velocity[i] = Eigen::Matrix<double, 3, 1>::Zero();
        Force[i] = Eigen::Matrix<double, 3, 1>::Zero();
    } 
    while(true){
        int TIMES = 25;
        for (int i = 0; i < TIMES; ++i)
        {
            for (int i = 0; i < meshV.rows(); ++i)
            {
                std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > tuple = ihla.applyForces(i, Velocity, Force, Position);
                Eigen::Matrix<double, 3, 1> vn = std::get<0>(tuple);
                Eigen::Matrix<double, 3, 1> xn = std::get<1>(tuple);
                //
                Velocity[i] = vn;
                Position[i] = xn; 
                // std::cout<<"i:"<<i<<", pos:( "<<xn[0]<<", "<<xn[1]<<", "<<xn[2]<<" )"<<std::endl;
            } 
            
            for (int i = 0; i < meshV.rows(); ++i)
            {
                std::tuple< Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1> > tuple = ihla.computeInternalForces(i, Velocity, Position);
                Eigen::Matrix<double, 3, 1> vn = std::get<0>(tuple);
                Eigen::Matrix<double, 3, 1> f = std::get<1>(tuple);
                //
                Velocity[i] = vn;
                Force[i] = f; 
            } 
        } 
        
        double min_diff = 1000;
        double max_diff = 0;
        for (int i = 0; i < meshV.rows(); ++i)
        {
            double norm = (Position[i]-OriginalPosition[i]).norm();
            if (norm < min_diff)
            {
                min_diff = norm;
            }
            if(norm > max_diff){
                max_diff = norm;
            } 
        } 
        std::cout<<"After updating, min_offset: "<<min_diff<<", max_offset: "<<max_diff<<std::endl;
    
        polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(Position);
    }
}


void myCallback()
{ 
    if (ImGui::Button("Start simulation")){ 
        update();
    } 
}

 

int main(int argc, const char * argv[]) {
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
    igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere3.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/libigl-polyscope-project/input/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/ddg-exercises/input/sphere.obj", meshV, meshF);
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/bumpy.off", meshV, meshF); // 69KB 5mins
    
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/cow.off", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    triangle_mesh.initialize(meshV, meshF);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::state::userCallback = myCallback;
    Eigen::Matrix<double, 3, 1> initV;
    initV << 0, 0, -100;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Position.push_back(meshV.row(i).transpose());
        OriginalPosition.push_back(meshV.row(i).transpose());
        //
        Velocity.push_back(Eigen::Matrix<double, 3, 1>::Zero());
        Force.push_back(Eigen::Matrix<double, 3, 1>::Zero());
    } 
    // for (int i = 0; i < meshV.rows(); ++i)
    // {
    //     std::cout<<"i: "<<i<<", ("<<Position[i][0]<<", "<<Position[i][1]<<", "<<Position[i][2]<<")"<<std::endl;
    // } 
    polyscope::show();
    return 0;
}
