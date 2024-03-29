#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include "Connectivity.h"
#include "nanoflann.hpp"

namespace iheartmesh {

struct MPoint {
  std::vector<Eigen::VectorXd> points;
  inline size_t kdtree_get_point_count() const { return points.size(); }
  inline double kdtree_get_pt(const size_t idx, int dim) const { return points[idx](dim); }
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& bb) const {
    return false;
  }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, MPoint>, MPoint, 3>
      MESH_KD_Tree_T;

class PointCloudWrapper {
public:
  PointCloudWrapper(std::vector<Eigen::VectorXd>& P);
  ~PointCloudWrapper();
  MPoint data;
  MESH_KD_Tree_T tree;
  std::vector<size_t> radiusSearch(Eigen::VectorXd query, double rad=0.1);
  std::vector<size_t> kNearest(Eigen::VectorXd query, size_t k);
  std::vector<size_t> kNearestNeighbors(size_t sourceInd, size_t k);
};

std::vector<std::vector<size_t>> GetPointNeighbors(std::vector<Eigen::VectorXd>& P, int k=10);
std::vector<std::vector<size_t>> GetPointNeighbors(std::vector<Eigen::VectorXd>& P, double rad=0.1);


class PointCloud: public Connectivity {
public:
    PointCloud();
    PointCloud(std::vector<Eigen::VectorXd>& P, std::vector<std::vector<size_t>> neighbors);
    void build_boundary_mat1(); // E -> V, size: |V|x|E|, boundary of edges 
    //
    // 
    bool numerical_order;       // whether the indices are stored as numerical order in edges/faces
};


}
