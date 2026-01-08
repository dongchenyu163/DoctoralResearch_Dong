#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

// ScoreCalculator implements Algorithms 2–4 in the specification:
// - filterByGeoScore → Algorithm 2 (GeoFilter)
// - calcPositionalScores / calcPositionalDistances → Algorithm 3 (PosScore)
// - calcDynamicsScores → Algorithm 4 (DynScore)
// Python orchestrates Algorithm 1 and provides data ownership via pcl::PointCloud.
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

  // Store Ω_low points/normals for later scoring. Expect points == normals rows == M.
  void setPointCloud(const Eigen::Ref<const PointMatrix>& points,
                     const Eigen::Ref<const PointMatrix>& normals);
  void configureLogging(const std::string& logger_name,
                        bool enable_console,
                        const std::string& file_path,
                        const std::string& level);

  void setMaxCandidates(std::int64_t max_candidates) noexcept { max_candidates_ = max_candidates; }
  // Geometry weights (Algorithm 2). Larger w_tbl prioritizes table clearance, etc.
  void setGeoWeights(double w_fin, double w_knf, double w_tbl) noexcept;
  // Ratio in (0,1] controlling how many combinations survive Algorithm 2.
  void setGeoFilterRatio(double ratio) noexcept;
  void setGeoRandomSeed(std::uint32_t seed) noexcept { geo_seed_ = seed; }
  void setDynamicsRandomSeed(std::uint32_t seed) noexcept { dyn_seed_ = seed; }

  // Algorithm 2: compute S_geo and select top rows.
  CandidateMatrix filterByGeoScore(const Eigen::Ref<const CandidateMatrix>& candidate_indices,
                                   const Eigen::Vector3d& knife_p,
                                   const Eigen::Vector3d& knife_n,
                                   double table_z) const;
  // Sorted row indices from the most recent GeoFilter call (descending score).
  Eigen::VectorXi lastGeoOrder() const;

  // Algorithm 3: E_pdir (smaller angle between PCA axis and knife normal is better).
  Eigen::VectorXd calcPositionalScores(const Eigen::Ref<const CandidateMatrix>& candidate_indices,
                                       const Eigen::Vector3d& knife_p,
                                       const Eigen::Vector3d& knife_n) const;

  // Algorithm 3: E_pdis distance-to-plane term (normalized per step).
  Eigen::VectorXd calcPositionalDistances(const Eigen::Ref<const CandidateMatrix>& candidate_indices,
                                          const Eigen::Vector3d& knife_p,
                                          const Eigen::Vector3d& knife_n) const;

  // Algorithm 4: evaluate combined dynamics score per candidate.
  // wrench: 6x1 knife wrench (fx,fy,fz,mx,my,mz).
  // center: object center (used to shift contact points for torque).
  // friction_coef: μ term; increasing it relaxes tangential limits.
  // friction_angle_deg: friction cone aperture (°); larger = wider cone.
  // max_attempts: number of force generation attempts per candidate.
  // balance_threshold: residual norm threshold for wrench balance.
  // force_min/max: range of sampled normal force magnitudes.
  // cone_angle_max_deg: max half-angle for sampling inside the cone.
  Eigen::VectorXd calcDynamicsScores(const Eigen::Ref<const CandidateMatrix>& candidate_indices,
                                     const Eigen::VectorXd& wrench,
                                     const Eigen::Vector3d& center,
                                     bool planar_constraint,
                                     double friction_coef,
                                     double friction_angle_deg,
                                     int max_attempts,
                                     double balance_threshold,
                                     double force_min,
                                     double force_max,
                                     double cone_angle_max_deg) const;
  double calcForceResidual(const Eigen::VectorXi& indices,
                           const Eigen::VectorXd& wrench,
                           const Eigen::Vector3d& center,
                           bool planar_constraint,
                           const Eigen::VectorXd& f) const;
  bool checkRandomForceBalance(const Eigen::VectorXi& indices,
                               const Eigen::VectorXd& wrench,
                               const Eigen::Vector3d& center,
                               bool planar_constraint,
                               double balance_threshold,
                               const Eigen::VectorXd& f_init =
                                   Eigen::VectorXd::Constant(1, std::numeric_limits<double>::quiet_NaN())) const;

  using ForceAttempt = std::tuple<Eigen::VectorXd, Eigen::VectorXd, double, double, double>;
  const std::vector<std::vector<ForceAttempt>>& lastDynamicsAttempts() const { return last_dyn_attempts_; }

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
  Eigen::Vector3d computeCentroid(const PointCloud& subset) const;
  Eigen::MatrixXd buildGraspMatrix(const Eigen::VectorXi& indices,
                                   const Eigen::Vector3d& center,
                                   bool planar_constraint) const;

  PointCloudPtr cloud_;
  pcl::KdTreeFLANN<PointT> kd_tree_;
  std::int64_t max_candidates_ = 0;
  double geo_ratio_ = 1.0;
  GeoWeights geo_weights_;
  std::uint32_t geo_seed_ = 42;
  std::uint32_t dyn_seed_ = 42;
  std::shared_ptr<spdlog::logger> core_logger_;
  std::shared_ptr<spdlog::logger> geo_logger_;
  std::shared_ptr<spdlog::logger> pos_logger_;
  std::shared_ptr<spdlog::logger> dyn_logger_;
  mutable Eigen::VectorXi last_geo_order_;
  mutable std::vector<std::vector<ForceAttempt>> last_dyn_attempts_;
};
