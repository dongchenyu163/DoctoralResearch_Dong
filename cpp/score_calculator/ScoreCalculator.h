#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class ScoreCalculator {
 public:
  using PointMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using CandidateMatrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using PointT = pcl::PointXYZRGBNormal;
  using PointCloud = pcl::PointCloud<PointT>;
  using PointCloudPtr = PointCloud::Ptr;

  struct GeoWeights {
    double w_fin = 1.0;
    double w_knf = 1.0;
    double w_tbl = 1.0;
  };

  ScoreCalculator() = default;

  void setPointCloud(const Eigen::Ref<const PointMatrix>& points,
                     const Eigen::Ref<const PointMatrix>& normals);

  void setMaxCandidates(std::int64_t max_candidates) noexcept { max_candidates_ = max_candidates; }
  void setGeoWeights(double w_fin, double w_knf, double w_tbl) noexcept;
  void setGeoFilterRatio(double ratio) noexcept;

  CandidateMatrix filterByGeoScore(const Eigen::Ref<const CandidateMatrix>& candidate_indices,
                                   const Eigen::Vector3d& knife_p,
                                   const Eigen::Vector3d& knife_n,
                                   double table_z) const;

  std::int64_t pointCount() const noexcept {
    return cloud_ ? static_cast<std::int64_t>(cloud_->size()) : 0;
  }

 private:
  struct RowScore {
    Eigen::Index row_index;
    double e_fin;
    double e_knf;
    double e_tbl;
  };

  double computeMinPairwiseDistance(const PointCloud& subset) const;

  PointCloudPtr cloud_;
  pcl::KdTreeFLANN<PointT> kd_tree_;
  std::int64_t max_candidates_ = 0;
  double geo_ratio_ = 1.0;
  GeoWeights geo_weights_;
};
