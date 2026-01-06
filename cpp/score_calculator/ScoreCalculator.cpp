#include "ScoreCalculator.h"

#include <stdexcept>

void ScoreCalculator::setPointCloud(const Eigen::Ref<const PointMatrix>& points,
                                    const Eigen::Ref<const PointMatrix>& normals) {
  if (points.rows() != normals.rows()) {
    throw std::invalid_argument("points and normals must have the same row count");
  }
  points_ = points;
  normals_ = normals;
}

ScoreCalculator::CandidateMatrix ScoreCalculator::filterByGeoScore(
    const Eigen::Ref<const CandidateMatrix>& candidate_indices,
    const Eigen::Vector3d& /*knife_p*/,
    const Eigen::Vector3d& /*knife_n*/,
    double /*table_z*/) const {
  if (candidate_indices.rows() == 0) {
    return CandidateMatrix(0, candidate_indices.cols());
  }
  Eigen::Index rows_to_copy = candidate_indices.rows();
  if (max_candidates_ > 0 && max_candidates_ < rows_to_copy) {
    rows_to_copy = static_cast<Eigen::Index>(max_candidates_);
  }
  return candidate_indices.topRows(rows_to_copy);
}
