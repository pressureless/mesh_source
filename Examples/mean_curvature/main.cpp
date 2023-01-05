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

    double t;
    double area(
        const int & f,
        const int & p,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    {
        const long dim_0 = x.size();
        double area_ret;
        // q,r = GetNeighborVerticesInFace(f, p)
        std::tuple< int, int > tuple = GetNeighborVerticesInFace(f, p);
        int q = std::get<0>(tuple);
        int r = std::get<1>(tuple);

        // pq = x_q - x_p
        Eigen::Matrix<double, 3, 1> pq = x.at(q) - x.at(p);

        // qr = x_r - x_q
        Eigen::Matrix<double, 3, 1> qr = x.at(r) - x.at(q);

        // pr = x_r - x_p
        Eigen::Matrix<double, 3, 1> pr = x.at(r) - x.at(p);

        // A = ½||(x_q-x_p)×(x_r-x_p)||
        double A = (1/double(2)) * (((x.at(q) - x.at(p))).cross((x.at(r) - x.at(p)))).lpNorm<2>();

        // dotp = pq ⋅ pr
        double dotp = (pq).dot(pr);

        // dotq = (x_q-x_r) ⋅ pq
        double dotq = ((x.at(q) - x.at(r))).dot(pq);

        // dotr = qr⋅ pr
        double dotr = (qr).dot(pr);

        // cotq = dotq/(2A)
        double cotq = dotq / double((2 * A));

        // cotr = dotr/(2A)
        double cotr = dotr / double((2 * A));
        if(A == 0){
            area_ret = 0;
        }
        else if(dotp < 0){
            area_ret = (1/double(2)) * A;
        }
        else if((dotq < 0) || (dotr < 0)){
            area_ret = (1/double(4)) * A;
        }
        else{
            area_ret = (1/double(4)) * (cotq * pow((pr).lpNorm<2>(), 2) + cotr * pow((pq).lpNorm<2>(), 2));
        }
        return area_ret;    
    }
    double cot(
        const int & k,
        const int & j,
        const int & i,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    {
        const long dim_1 = x.size();
        double cot_ret;
        // oj, oi = OrientedVertices(k, j, i)
        std::tuple< int, int > tuple_1 = OrientedVertices(k, j, i);
        int oj = std::get<0>(tuple_1);
        int oi = std::get<1>(tuple_1);

        // cos = (x_oj - x_k)⋅(x_oi-x_k)
        double cos = ((x.at(oj) - x.at(k))).dot((x.at(oi) - x.at(k)));

        // sin = ||(x_oj - x_k)×(x_oi-x_k)||
        double sin = (((x.at(oj) - x.at(k))).cross((x.at(oi) - x.at(k)))).lpNorm<2>();
        if(sin != 0){
            cot_ret = cos / double(sin);
        }
        else{
            cot_ret = 0;
        }
        return cot_ret;    
    }
    Eigen::Matrix<double, 3, 1> Ax(
        const int & i,
        const std::vector<Eigen::Matrix<double, 3, 1>> & x)
    {
        const long dim_2 = x.size();
        double sum_0 = 0;
        for(int f : FaceNeighbors(i)){
            sum_0 += area(f, i, x);
        }
        // A = (sum_(f ∈ FaceNeighbors(i)) area(f, i, x))
        double A = (sum_0);

        Eigen::MatrixXd sum_1 = Eigen::MatrixXd::Zero(3, 1);
        for(int j : VertexOneRing(i)){
                // k, l = OppositeVertices(GetEdgeIndex(i,j))
            std::tuple< int, int > tuple_2 = OppositeVertices(GetEdgeIndex(i, j));
            int k = std::get<0>(tuple_2);
            int l = std::get<1>(tuple_2);
                // `cot(α)` = cot(k, j, i, x)
            double cotα = cot(k, j, i, x);
                // `cot(β)` = cot(l, i, j, x)
            double cotβ = cot(l, i, j, x);
            sum_1 += (cotα + cotβ) * (x.at(j) - x.at(i));
        }
        // K = 1/(2A)(sum_(j ∈ VertexOneRing(i)) (`cot(α)` + `cot(β)`)(x_j - x_i) 
    // where k, l = OppositeVertices(GetEdgeIndex(i,j)),
    // `cot(α)` = cot(k, j, i, x),
    // `cot(β)` = cot(l, i, j, x) )
        Eigen::Matrix<double, 3, 1> K = 1 / double((2 * A)) * (sum_1);
        return x.at(i) - t * K;    
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
        const double & t)
    :
    _FundamentalMeshAccessors(M)
    {
        int dimv_0 = M.n_vertices();
        int dime_0 = M.n_edges();
        int dimf_0 = M.n_faces();
        this->t = t;
    
    }
};

Eigen::MatrixXd meshV;
Eigen::MatrixXi meshF;
TriangleMesh triangle_mesh;

double cg_tolerance = 1e-5;
int MAX_ITERATION = 1;
double step = 10;

std::vector<Eigen::Matrix<double, 3, 1>> OriginalPosition;
std::vector<Eigen::Matrix<double, 3, 1>> Position;

void axpy3(const std::vector<Eigen::Matrix<double, 3, 1>>& X,
           const double                            alpha,
           const double                            beta,
           std::vector<Eigen::Matrix<double, 3, 1>>&       Y)
{
    // Y = beta*Y + alpha*X 
    int size = static_cast<int>(X.size()); 
    for (int i = 0; i < size; ++i) {
        Y[i] *= beta;
        Y[i] += alpha * X[i];
    }
}

double dot3(const std::vector<Eigen::Matrix<double, 3, 1>>& A,
       const std::vector<Eigen::Matrix<double, 3, 1>>& B)
{

    double  ret = 0;
    int size = static_cast<int>(A.size());
    for (int i = 0; i < size; ++i) {
        ret += A[i].dot(B[i]);
    }
    return ret;
}


void cg(std::vector<Eigen::Matrix<double, 3, 1>>& X,
        std::vector<Eigen::Matrix<double, 3, 1>>& B,
        std::vector<Eigen::Matrix<double, 3, 1>>& R,
        std::vector<Eigen::Matrix<double, 3, 1>>& P,
        std::vector<Eigen::Matrix<double, 3, 1>>& S,
        double& start_residual,
        double&  stop_residual,
        iheartla& ihla)
{
    // Conjugate Gradients
    // Page 50 in "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"
    // CG solver. Solve for the three coordinates simultaneously

    // s = Ax 
    for (int i = 0; i < meshV.rows(); ++i)
    {
        S[i] = ihla.Ax(i, X);
    } 
 
    // r = b - s = b - Ax
    // p = r 
    for (int i = 0; i < int(meshV.rows()); ++i) {
        R[i] = B[i] - S[i];
        R[i] = B[i] - S[i];
        R[i] = B[i] - S[i];

        P[i] = R[i];
        P[i] = R[i];
        P[i] = R[i];
    }
 
    // delta_new = <r,r>
    double delta_new = dot3(R, R);

    // delta_0 = delta_new
    double delta_0 = delta_new;
        std::cout<<"delta_new delta_new: "<<delta_new<<std::endl;

    start_residual = delta_0;
    uint32_t iter  = 0;
    while (iter < MAX_ITERATION) { 
        // s = Ap 
        for (int i = 0; i < meshV.rows(); ++i)
        {
            S[i] = ihla.Ax(i, P);
        }  

        // alpha = delta_new / <s,p>
        double alpha = dot3(S, P);
        std::cout<<"alpha before: "<<alpha<<std::endl;
        alpha = delta_new / alpha;

        std::cout<<"alpha: "<<alpha<<std::endl;
        // x =  x + alpha*p
        axpy3(P, alpha, 1.0, X);

        // r = r - alpha*s
        axpy3(S, -alpha, 1.0, R);

        // delta_old = delta_new
        double delta_old(delta_new);

        // delta_new = <r,r>
        delta_new = dot3(R, R);

        // beta = delta_new/delta_old
        double beta(delta_new / delta_old);

        // exit if error is getting too low across three coordinates
        std::cout<<"delta_new:"<<delta_new<<std::endl;
        std::cout<<"cg_tolerance * cg_tolerance * delta_0:"<<cg_tolerance * cg_tolerance * delta_0<<std::endl;
            
        if (delta_new < cg_tolerance * cg_tolerance * delta_0) {
            break;
        }

        // p = beta*p + r
        axpy3(R, 1.0, beta, P);

        ++iter;
        std::cout<<"iter:"<<iter<<std::endl;
    }  
    stop_residual = delta_new;
}

void update(){
    iheartla ihla(triangle_mesh, step); 
    std::cout<<"After"<<std::endl;
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout<<"i: "<<i<<", "<<P[i]<<std::endl;
    // } 
    double start_residual, stop_residual;
    std::vector<Eigen::Matrix<double, 3, 1>> X(meshV.rows()), B(meshV.rows()), R(meshV.rows()), P(meshV.rows()), S(meshV.rows());

    for (int i = 0; i < meshV.rows(); ++i)
    {
        X[i] = Position[i];
        B[i] = Position[i];
    } 
    cg(X, B, R, P, S, start_residual, stop_residual, ihla);
    std::cout<<"start_residual:"<<start_residual<<", stop_residual:"<<stop_residual<<std::endl;
    
    Position = X;
    double min_diff = 1000;
    double max_diff = 0;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        double norm = (X[i]-OriginalPosition[i]).norm();
        if (norm < min_diff)
        {
            min_diff = norm;
        }
        if(norm > max_diff){
            max_diff = norm;
        }
        // std::cout<<"i: "<<i<<", ("<<X[i][0]-OriginalPosition[i][0]<<", "<<X[i][1]-OriginalPosition[i][1]<<", "<<X[i][2]-OriginalPosition[i][2]<<")"<<std::endl;
    } 
    std::cout<<"After updating, min_offset: "<<min_diff<<", max_offset: "<<max_diff<<std::endl;
    polyscope::getSurfaceMesh("my mesh")->updateVertexPositions(Position);
}


void myCallback()
{ 
    if (ImGui::Button("One step")){ 
        update();
    } 
    if (ImGui::Button("Ten steps")){
        for (int i = 0; i < 10; ++i)
        {
            update();
        }
    } 
    if (ImGui::Button("Fifty steps")){
        for (int i = 0; i < 50; ++i)
        {
            update();
        }
    } 
}

 

int main(int argc, const char * argv[]) {
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/cube.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/small_bunny.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere3.obj", meshV, meshF);
    igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOFF("/Users/pressure/Downloads/Laplacian-Mesh-Smoothing/Models/bumpy.off", meshV, meshF); // 69KB 5mins
    
    // igl::readOBJ("/Users/pressure/Downloads/mesh_source/models/sphere.obj", meshV, meshF);
    // igl::readOBJ("/Users/pressure/Documents/git/meshtaichi/vertex_normal/models/bunny.obj", meshV, meshF);
    // Initialize triangle mesh
    triangle_mesh.initialize(meshV, meshF);
    // Initialize polyscope
    polyscope::init();  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);
    polyscope::state::userCallback = myCallback;
    for (int i = 0; i < meshV.rows(); ++i)
    {
        Position.push_back(meshV.row(i).transpose());
        OriginalPosition.push_back(meshV.row(i).transpose());
    } 
    // for (int i = 0; i < meshV.rows(); ++i)
    // {
    //     std::cout<<"i: "<<i<<", ("<<Position[i][0]<<", "<<Position[i][1]<<", "<<Position[i][2]<<")"<<std::endl;
    // } 
    polyscope::show();
    return 0;
}
