#include "MeshHelper.h"

std::tuple<std::set<int>, std::set<int>, std::set<int>> MeshSets(const TriangleMesh& mesh){
	return std::tuple<std::set<int>, std::set<int>, std::set<int>>(mesh.Vi, mesh.Ei, mesh.Fi);
}

std::tuple<Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> > BoundaryMatrices(const TriangleMesh& mesh){
	return std::tuple< Eigen::SparseMatrix<int>, Eigen::SparseMatrix<int> >(mesh.bm1, mesh.bm2);
}

// std::set<int> vector_to_vertices(const TriangleMesh& mesh, const Eigen::VectorXi& vi){
// 	return mesh.vector_to_vertices(vi);
// }

// std::set<int> vector_to_edges(const TriangleMesh& mesh, const Eigen::VectorXi& ei){
// 	return mesh.vector_to_vertices(ei);
// }

// std::set<int> vector_to_faces(const TriangleMesh& mesh, const Eigen::VectorXi& fi){
// 	return mesh.vector_to_vertices(fi);
// }

std::set<int> nonzeros(Eigen::SparseMatrix<int> target){
    return nonzeros(target, true);
}

std::set<int> nonzeros(Eigen::SparseMatrix<int> target, bool is_row){
    std::set<int> result;
    for (int k=0; k<target.outerSize(); ++k){
      for (SparseMatrix<int>::InnerIterator it(target,k); it; ++it) {
        if (is_row)
        {
            result.insert(it.row());
        }
        else{
            result.insert(it.col());
        }
      }
    } 
    return result;
}
std::set<int> ValueSet(Eigen::SparseMatrix<int> target, int value){
    // return row indices for specific value
    return ValueSet(target, value, true);
}
std::set<int> ValueSet(Eigen::SparseMatrix<int> target, int value, bool is_row){
    std::set<int> result;
    for (int k=0; k<target.outerSize(); ++k){
      for (SparseMatrix<int>::InnerIterator it(target,k); it; ++it) {
        if (it.value() == value)
        {
            if (is_row)
            {
                result.insert(it.row());
            }
            else{
                result.insert(it.col());
            }
        }
      }
    } 
    return result;
}