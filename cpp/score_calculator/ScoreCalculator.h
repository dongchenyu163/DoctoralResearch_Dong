#pragma once

#include <cstdint>

#include <Eigen/Core>

class ScoreCalculator {
 public:
  using PointMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using CandidateMatrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  ScoreCalculator() = default;

  void setPointCloud(const Eigen::Ref<const PointMatrix>& points,
                     const Eigen::Ref<const PointMatrix>& normals);

  void setMaxCandidates(std::int64_t max_candidates) noexcept { max_candidates_ = max_candidates; }

  CandidateMatrix filterByGeoScore(const Eigen::Ref<const CandidateMatrix>& candidate_indices,
                                   const Eigen::Vector3d& knife_p,
                                   const Eigen::Vector3d& knife_n,
                                   double table_z) const;

  std::int64_t pointCount() const noexcept { return static_cast<std::int64_t>(points_.rows()); }

  const PointMatrix& points() const noexcept { return points_; }
  const PointMatrix& normals() const noexcept { return normals_; }

 private:
  PointMatrix points_;
  PointMatrix normals_;
  std::int64_t max_candidates_ = 0;
};
