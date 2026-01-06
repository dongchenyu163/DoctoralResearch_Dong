#include "ScoreCalculator.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

void ScoreCalculator::setPointCloud(const Eigen::Ref<const PointMatrix>& points,
                                    const Eigen::Ref<const PointMatrix>& normals) {
  if (points.rows() != normals.rows()) {
    throw std::invalid_argument("points and normals must have the same row count");
  }
  points_ = points;
  normals_ = normals;
}

void ScoreCalculator::setGeoWeights(double w_fin, double w_knf, double w_tbl) noexcept {
  geo_weights_.w_fin = w_fin;
  geo_weights_.w_knf = w_knf;
  geo_weights_.w_tbl = w_tbl;
}

void ScoreCalculator::setGeoFilterRatio(double ratio) noexcept {
  geo_ratio_ = std::clamp(ratio, 0.0, 1.0);
}

namespace {

double distanceToPlane(const Eigen::Vector3d& point,
                       const Eigen::Vector3d& plane_point,
                       const Eigen::Vector3d& plane_normal) {
  const double norm = plane_normal.norm();
  if (norm < 1e-12) {
    return 0.0;
  }
  return std::abs((point - plane_point).dot(plane_normal) / norm);
}

double minPairwiseDistance(const Eigen::Matrix<double, Eigen::Dynamic, 3>& pts) {
  if (pts.rows() < 2) {
    return 0.0;
  }
  double min_dist = std::numeric_limits<double>::max();
  for (Eigen::Index i = 0; i < pts.rows(); ++i) {
    for (Eigen::Index j = i + 1; j < pts.rows(); ++j) {
      const double dist = (pts.row(i) - pts.row(j)).norm();
      if (dist < min_dist) {
        min_dist = dist;
      }
    }
  }
  if (!std::isfinite(min_dist) || min_dist == std::numeric_limits<double>::max()) {
    return 0.0;
  }
  return min_dist;
}

double minTableDistance(const Eigen::Matrix<double, Eigen::Dynamic, 3>& pts, double table_z) {
  double min_dist = std::numeric_limits<double>::max();
  for (Eigen::Index i = 0; i < pts.rows(); ++i) {
    const double dist = pts(i, 2) - table_z;
    if (dist < min_dist) {
      min_dist = dist;
    }
  }
  if (!std::isfinite(min_dist) || min_dist == std::numeric_limits<double>::max()) {
    return 0.0;
  }
  return std::max(min_dist, 0.0);
}

}  // namespace

ScoreCalculator::CandidateMatrix ScoreCalculator::filterByGeoScore(
    const Eigen::Ref<const CandidateMatrix>& candidate_indices,
    const Eigen::Vector3d& knife_p,
    const Eigen::Vector3d& knife_n,
    double table_z) const {
  if (candidate_indices.rows() == 0 || candidate_indices.cols() == 0 || points_.rows() == 0) {
    return CandidateMatrix(0, candidate_indices.cols());
  }

  std::vector<RowScore> row_scores;
  row_scores.reserve(static_cast<std::size_t>(candidate_indices.rows()));

  Eigen::Matrix<double, Eigen::Dynamic, 3> buffer(candidate_indices.cols(), 3);

  for (Eigen::Index row = 0; row < candidate_indices.rows(); ++row) {
    bool row_valid = true;
    for (Eigen::Index col = 0; col < candidate_indices.cols(); ++col) {
      const int idx = candidate_indices(row, col);
      if (idx < 0 || idx >= points_.rows()) {
        row_valid = false;
        break;
      }
      buffer.row(col) = points_.row(idx);
    }
    if (!row_valid) {
      continue;
    }
    const double e_fin = minPairwiseDistance(buffer);
    const double e_knf = distanceToPlane(buffer.colwise().mean(), knife_p, knife_n);
    const double e_tbl = minTableDistance(buffer, table_z);
    row_scores.push_back({row, e_fin, e_knf, e_tbl});
  }

  if (row_scores.empty()) {
    return CandidateMatrix(0, candidate_indices.cols());
  }

  const Eigen::Index feature_rows = static_cast<Eigen::Index>(row_scores.size());
  Eigen::ArrayXd e_fin(feature_rows);
  Eigen::ArrayXd e_knf(feature_rows);
  Eigen::ArrayXd e_tbl(feature_rows);
  for (Eigen::Index i = 0; i < feature_rows; ++i) {
    e_fin(i) = row_scores[static_cast<std::size_t>(i)].e_fin;
    e_knf(i) = row_scores[static_cast<std::size_t>(i)].e_knf;
    e_tbl(i) = row_scores[static_cast<std::size_t>(i)].e_tbl;
  }

  auto normalize = [](Eigen::ArrayXd& arr) {
    const double min_v = arr.minCoeff();
    const double max_v = arr.maxCoeff();
    const double range = max_v - min_v;
    if (!std::isfinite(range) || range < 1e-12) {
      arr.setZero();
      return;
    }
    arr = (arr - min_v) / range;
  };

  normalize(e_fin);
  normalize(e_knf);
  normalize(e_tbl);

  Eigen::ArrayXd total = geo_weights_.w_fin * e_fin + geo_weights_.w_knf * e_knf + geo_weights_.w_tbl * e_tbl;

  std::vector<std::pair<double, Eigen::Index>> order;
  order.reserve(static_cast<std::size_t>(feature_rows));
  for (Eigen::Index i = 0; i < feature_rows; ++i) {
    order.emplace_back(total(i), row_scores[static_cast<std::size_t>(i)].row_index);
  }

  std::sort(order.begin(), order.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first > rhs.first;
  });

  Eigen::Index keep = static_cast<Eigen::Index>(order.size());
  if (geo_ratio_ > 0.0 && geo_ratio_ < 1.0) {
    keep = std::max<Eigen::Index>(1, static_cast<Eigen::Index>(std::round(geo_ratio_ * keep)));
  }
  if (max_candidates_ > 0 && max_candidates_ < keep) {
    keep = static_cast<Eigen::Index>(max_candidates_);
  }

  CandidateMatrix output(keep, candidate_indices.cols());
  for (Eigen::Index i = 0; i < keep; ++i) {
    output.row(i) = candidate_indices.row(order[static_cast<std::size_t>(i)].second);
  }
  return output;
}
