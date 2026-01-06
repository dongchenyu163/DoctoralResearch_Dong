#pragma once

#include <cstdint>

#include <Eigen/Core>

class ScoreCalculator {
 public:
  using PointMatrix = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

  ScoreCalculator() = default;

  void setPointCloud(const Eigen::Ref<const PointMatrix>& points,
                     const Eigen::Ref<const PointMatrix>& normals);

  std::int64_t pointCount() const noexcept { return static_cast<std::int64_t>(points_.rows()); }

  const PointMatrix& points() const noexcept { return points_; }
  const PointMatrix& normals() const noexcept { return normals_; }

 private:
  PointMatrix points_;
  PointMatrix normals_;
};
