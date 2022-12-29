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
#include "MeshHelper.h"
#include "dec_util.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

using Eigen::MatrixXi;
using Eigen::MatrixXd;
/*
arctan from trigonometry
nEi as E, get_vertices_e as endpoints, get_vertices_f as vof, get_diamond_faces_e as foe  from triangle_mesh( x, Faces )
δ ∈ ℝ
x ∈ ℝ^(n×3)
x̃ ∈ ℝ^(n×3)
Faces ∈ ℤ^(m×3)

l(e, x) = ‖x_i,* - x_j,*‖ where e ∈ ℤ,x ∈ ℝ^(n×3) ; i,j = endpoints(e)

A(f, x) = ½‖(x_j,*- x_i,*)×(x_k,*-x_i,*)‖ where f ∈ ℤ,x ∈ ℝ^(n×3) ; i,j,k = vof(f)

N(f, x) = ((x_j,*- x_i,*)×(x_k,*-x_i,*))/(2A(f,x)) where f ∈ ℤ,x ∈ ℝ^(n×3) ; i,j,k = vof(f)

D(e, x) = ⅓(A(f, x) + A(`f'`,x) )where e ∈ ℤ,x ∈ ℝ^(n×3) ; f, `f'` = foe(e)

θ(e, x) = arctan(((x_j,*- x_i,*)⋅(N(f,x) × N(`f'`,x))) / (N(f,x) ⋅ N(`f'`,x)) ) where e ∈ ℤ,x ∈ ℝ^(n×3) ; f, `f'` = foe(e); i,j = endpoints(e)

Wbend = δ³ sum_(e ∈ E) (θ(e, x) - θ(e, x̃))² D(e, x)⁻¹ l(e, x)²
*/
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <set>
#include "TriangleMesh.h"

struct iheartla {

    std::vector<Eigen::Matrix<double, 3, 1>> x;
    Eigen::Matrix<double, 3, 1> getVertexNormal(
        const int & i)
    {
        Eigen::MatrixXd sum_0 = Eigen::MatrixXd::Zero(3, 1);
        for(int f : N(i)){
                // j, k = getNeighborVerticesInFace(f, i)
            std::tuple< int, int > tuple = getNeighborVerticesInFace(f, i);
            int j = std::get<0>(tuple);
            int k = std::get<1>(tuple);
                // n = (x_j- x_i)×(x_k-x_i)
            Eigen::Matrix<double, 3, 1> n = ((x.at(j) - x.at(i))).cross((x.at(k) - x.at(i)));
            sum_0 += n / double((pow((x.at(j) - x.at(i)).lpNorm<2>(), 2) * pow((x.at(k) - x.at(i)).lpNorm<2>(), 2)));
        }
        return (sum_0);    
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
        Eigen::SparseMatrix<int> Vertices_0(
            const int & f)
        {
            std::set<int > Vertices_0set_0({f});
            return uve * uef * M.faces_to_vector(Vertices_0set_0);    
        }
        Eigen::SparseMatrix<int> Vertices_1(
            const std::set<int > & f)
        {
            return uve * uef * M.faces_to_vector(f);    
        }
        Eigen::SparseMatrix<int> Vertices_2(
            const int & e)
        {
            std::set<int > Vertices_2set_0({e});
            return uve * M.edges_to_vector(Vertices_2set_0);    
        }
        Eigen::SparseMatrix<int> Vertices_3(
            const std::set<int > & e)
        {
            return uve * M.edges_to_vector(e);    
        }
        std::set<int > Vertices_4(
            const int & v)
        {
            std::set<int > Vertices_4set_0({v});
            return nonzeros(uve * uve.transpose() * M.vertices_to_vector(Vertices_4set_0));    
        }
        Eigen::SparseMatrix<int> Vertices_5(
            const std::set<int > & v)
        {
            return uve * uve.transpose() * M.vertices_to_vector(v);    
        }
        std::set<int > Faces(
            const int & v)
        {
            std::set<int > Facesset_0({v});
            return nonzeros((uve * uef).transpose() * M.vertices_to_vector(Facesset_0));    
        }
        std::tuple< int, int > getNeighborVerticesInFace(
            const int & f,
            const int & v)
        {
            std::set<int > getNeighborVerticesInFaceset_0({f});
            // es = edgeset(NonZeros(ef IndicatorVector(M, {f})))
            std::set<int > es = nonzeros(ef * M.faces_to_vector(getNeighborVerticesInFaceset_0));
            std::set<int > getNeighborVerticesInFaceset_1;
            for(int s : es){
                if(ve.coeff(v, s) != 0){
                    getNeighborVerticesInFaceset_1.insert(s);
                }
            }
            // nes = { s for s ∈ es if ve_v,s != 0 }
            std::set<int > nes = getNeighborVerticesInFaceset_1;
            std::set<int > getNeighborVerticesInFaceset_2;
            for(int e : nes){
                if(ef.coeff(e, f) * ve.coeff(v, e) == -1){
                    getNeighborVerticesInFaceset_2.insert(e);
                }
            }
            std::vector<int> stdv(getNeighborVerticesInFaceset_2.begin(), getNeighborVerticesInFaceset_2.end());
            Eigen::VectorXi vec(Eigen::Map<Eigen::VectorXi>(&stdv[0], stdv.size()));
            // vvec1 = vec({ e for e ∈ nes if ef_e,f ve_v,e == -1})
            Eigen::VectorXi vvec1 = vec;
            std::set<int > getNeighborVerticesInFaceset_3;
            for(int e : nes){
                if(ef.coeff(e, f) * ve.coeff(v, e) == 1){
                    getNeighborVerticesInFaceset_3.insert(e);
                }
            }
            std::vector<int> stdv_1(getNeighborVerticesInFaceset_3.begin(), getNeighborVerticesInFaceset_3.end());
            Eigen::VectorXi vec_1(Eigen::Map<Eigen::VectorXi>(&stdv_1[0], stdv_1.size()));
            // vvec2 = vec({ e for e ∈ nes if ef_e,f ve_v,e == 1 })
            Eigen::VectorXi vvec2 = vec_1;
            return std::tuple<int,int >{ vvec1[1-1],vvec2[1-1] };    
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
    std::tuple< int, int > getNeighborVerticesInFace(int p0,int p1){
        return _FundamentalMeshAccessors.getNeighborVerticesInFace(p0,p1);
    };
    std::set<int > N(int p0){
        return _FundamentalMeshAccessors.Faces(p0);
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




int main(int argc, const char * argv[]) {
    //       2
    //     / | \
    //    0  |  3
    //     \ | /
    //       1
    Eigen::Matrix<double, Eigen::Dynamic, 3> P(4, 3);
    P <<
    0, 1, 0,
    1, 0, 0,
    1, 1, 0,
    2, 1, 0;
    std::vector<Eigen::Matrix<double, 3, 1>> x;
    x.push_back(P.row(0).transpose());
    x.push_back(P.row(1).transpose());
    x.push_back(P.row(2).transpose());
    x.push_back(P.row(3).transpose());
    Eigen::Matrix<double, Eigen::Dynamic, 3> P1(4, 3);
    P1 <<
    1, 1, 1,
    1, 2, 3,
    2, 1, 4,
    3, 1, 2;
    MatrixXi Face(2,3);
    Face <<
    0,1,2,
    2,1,3; 
    MatrixXi Tet(2,4);
    Tet <<
    0,1,2,3,
    1,2,3,4; 
    TriangleMesh triangle_mesh_0;
    triangle_mesh_0.initialize(P, Face);

    // std::cout <<"triangle_mesh_0.bm1:\n"<< triangle_mesh_0.bm1<< std::endl;
    // std::cout <<"triangle_mesh_0.bm2:\n"<< triangle_mesh_0.bm2 << std::endl;
    // std::cout <<"original:\n"<< triangle_mesh_0.bm1 * triangle_mesh_0.bm2 << std::endl;
    iheartla ihla(triangle_mesh_0, x);
    // std::cout<<"ihla, v1:"<<ihla.v1<<std::endl;
    // std::cout<<"ihla, v2:"<<ihla.v2<<std::endl;
    Eigen::Matrix<double, 3, 1> a = ihla.getVertexNormal(1);
    std::cout<<"a:"<<a<<std::endl;
    // TriangleMesh dec(V, T);
    // std::tuple< int, int > res = dec.get_diamond_vertices_e(2, 1);
    // std::cout<<"edges:"<<dec.edges<<std::endl;
    // for (std::set<int>::iterator it = ii.triangle_mesh_0.nEi.begin(); it != ii.triangle_mesh_0.nEi.end(); ++it) {
    //         std::cout << *it << ", ";
    //     }
    // std::cout<<std::endl;
    // std::cout<<"d: "<<ii.a<<std::endl;
    // std::cout<<"second_face:"<<std::get<1>(res)<<std::endl;
    // insert code here... 

    Eigen::MatrixXd meshV;
    Eigen::MatrixXi meshF;
    igl::readOBJ("/Users/pressure/Documents/git/mesh_source/models/bunny.obj", meshV, meshF);


    // Initialize polyscope
    polyscope::init(); 

    // Register a surface mesh structure
    // `meshVerts` is a Vx3 array-like container of vertex positions
    // `meshFaces` is a Fx3 array-like container of face indices  
    polyscope::registerSurfaceMesh("my mesh", meshV, meshF);

    // Add a scalar and a vector function defined on the mesh
    // `scalarQuantity` is a length V array-like container of values
    // `vectorQuantity` is an Fx3 array-like container of vectors per face
    // polyscope::getSurfaceMesh("my mesh")->addVertexScalarQuantity("my_scalar", scalarQuantity);
    // polyscope::getSurfaceMesh("my mesh")->addFaceVectorQuantity("my_vector", vectorQuantity);

    // View the point cloud and mesh we just registered in the 3D UI
    polyscope::show();

    return 0;
}
