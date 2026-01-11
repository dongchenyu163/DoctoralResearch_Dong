#include "ScoreCalculator.h"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <random>
// #include <spdlog/spdlog.h>
#include <stdexcept>

#include <pcl/visualization/pcl_visualizer.h>

namespace {

spdlog::level::level_enum ParseLevel(const std::string& level) {
  try {
    return spdlog::level::from_str(level);
  } catch (const spdlog::spdlog_ex&) {
    return spdlog::level::info;
  }
}

}  // namespace

Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& mat, double tol) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& singular = svd.singularValues();
  double max_singular = singular.size() ? singular.maxCoeff() : 0.0;
  double threshold = tol * std::max(mat.rows(), mat.cols()) * max_singular;
  Eigen::VectorXd inv = singular;
  for (Eigen::Index i = 0; i < singular.size(); ++i) {
    inv(i) = singular(i) > threshold ? 1.0 / singular(i) : 0.0;
  }
  return svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose();
}

Eigen::Vector3d SafeNormal(const ScoreCalculator::PointT& point) {
  Eigen::Vector3d normal(static_cast<double>(point.normal_x),
                         static_cast<double>(point.normal_y),
                         static_cast<double>(point.normal_z));
  double norm = normal.norm();
  if (norm < 1e-6) {
    return Eigen::Vector3d(0.0, 1.0, 0.0);
  }
  return normal / norm;
}

Eigen::Vector3d SampleForceInCone(const Eigen::Vector3d& normal,
                                  std::mt19937& rng,
                                  std::uniform_real_distribution<double>& angle_dist,
                                  std::uniform_real_distribution<double>& normal_dist)
{
  // --- 0) normalize cone axis ---
  Eigen::Vector3d n = normal.normalized();

  // --- 1) build orthonormal basis (t1, t2, n) ---
  Eigen::Vector3d axis =
      (std::abs(n.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  Eigen::Vector3d t1 = n.cross(axis);
  if (t1.squaredNorm() < 1e-18) {
    axis = Eigen::Vector3d::UnitY();
    t1 = n.cross(axis);
  }
  t1.normalize();
  Eigen::Vector3d t2 = n.cross(t1); // already unit

  // --- 2) sample normal magnitude ---
  const double Fn = normal_dist(rng);

  // --- 3) sample theta uniformly in *area* (uniform in cos(theta)) ---
  // angle_dist is assumed to be U[0, theta_max]
  const double theta_max = angle_dist.b();
  const double cos_theta_max = std::cos(theta_max);

  Eigen::Vector3d dir = Eigen::Vector3d::Zero();
  double scale = 0.0;
  do {
    // reuse angle_dist as a [0,1] random source
    const double u =
      (angle_dist(rng) - angle_dist.a()) / (angle_dist.b() - angle_dist.a());
      const double cos_theta = (1.0 - u) * 1.0 + u * cos_theta_max;
      const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

  // --- 4) azimuth phi: internal static uniform [0, 2pi) ---
  static std::uniform_real_distribution<double> phi_dist(0.0, 2.0 * M_PI);
  const double phi = phi_dist(rng);
  
  // --- 5) direction inside cone ---
  dir = cos_theta * n + sin_theta * (std::cos(phi) * t1 + std::sin(phi) * t2);

  scale = (cos_theta > 1e-12) ? (Fn / cos_theta) : 0.0;
  
    dir.normalize();
    n.normalize();
  }while (dir.dot(n) < cos_theta_max);  // Reject samples outside cone

  // --- 6) scale so that (force · n) == Fn ---
  // since dir·n == cos(theta)
  // scale = (cos_theta > 1e-12) ? (Fn / cos_theta) : 0.0;
  return dir * scale;
}

Eigen::VectorXd ReducePlanarWrench(const Eigen::VectorXd& wrench) {
  Eigen::VectorXd reduced(3);
  reduced(0) = wrench(0);
  reduced(1) = wrench(1);
  reduced(2) = wrench(5);
  return reduced;
}

Eigen::VectorXd ExpandPlanarResidual(const Eigen::VectorXd& residual) {
  Eigen::VectorXd expanded(6);
  expanded.setZero();
  expanded(0) = residual(0);
  expanded(1) = residual(1);
  expanded(5) = residual(2);
  return expanded;
}

Eigen::MatrixXd ReducePlanarGraspMatrix(const Eigen::MatrixXd& grasp) {
  Eigen::MatrixXd reduced(3, grasp.cols());
  reduced.row(0) = grasp.row(0);
  reduced.row(1) = grasp.row(1);
  reduced.row(2) = grasp.row(5);
  return reduced;
}

// Store Ω_low points inside a PCL cloud so every algorithm reuses identical data.
void ScoreCalculator::setPointCloud(const Eigen::Ref<const PointMatrix>& points,
                                    const Eigen::Ref<const PointMatrix>& normals) {
  if (points.rows() != normals.rows()) {
    throw std::invalid_argument("points and normals must have the same row count");
  }
  cloud_.reset(new PointCloud());
  cloud_->reserve(points.rows());
  for (Eigen::Index i = 0; i < points.rows(); ++i) {
    PointT p;
    p.x = static_cast<float>(points(i, 0));
    p.y = static_cast<float>(points(i, 1));
    p.z = static_cast<float>(points(i, 2));
    p.normal_x = static_cast<float>(normals(i, 0));
    p.normal_y = static_cast<float>(normals(i, 1));
    p.normal_z = static_cast<float>(normals(i, 2));
    cloud_->push_back(p);
  }
  cloud_->width = static_cast<uint32_t>(cloud_->size());
  cloud_->height = 1;
  cloud_->is_dense = true;
  kd_tree_.setInputCloud(cloud_);
  if (core_logger_) {
    SPDLOG_LOGGER_INFO(core_logger_, "Loaded point cloud with {} samples", cloud_->size());
  }
}

void ScoreCalculator::setGeoWeights(double w_fin, double w_knf, double w_tbl) noexcept {
  geo_weights_.w_fin = w_fin;
  geo_weights_.w_knf = w_knf;
  geo_weights_.w_tbl = w_tbl;
  if (core_logger_) {
    SPDLOG_LOGGER_DEBUG(core_logger_, "Geo weights updated to {}, {}, {}", w_fin, w_knf, w_tbl);
  }
}

void ScoreCalculator::setForceWeights(double w_mag, double w_dir, double w_var) noexcept {
  force_weights_.w_mag = w_mag;
  force_weights_.w_dir = w_dir;
  force_weights_.w_var = w_var;
  if (core_logger_) {
    SPDLOG_LOGGER_DEBUG(core_logger_, "Force weights updated to {}, {}, {}", w_mag, w_dir, w_var);
  }
}

void ScoreCalculator::setGeoFilterRatio(double ratio) noexcept {
  geo_ratio_ = std::clamp(ratio, 0.0, 1.0);
  if (core_logger_) {
    SPDLOG_LOGGER_DEBUG(core_logger_, "Geo filter ratio set to {}", geo_ratio_);
  }
}

void ScoreCalculator::configureLogging(const std::string& logger_name,
                                       bool enable_console,
                                       const std::string& file_path,
                                       const std::string& level) {
  std::vector<spdlog::sink_ptr> sinks;
  if (enable_console) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("[%m%d%H%M%S_%e] CPP %n %l: %v :: %!");
    sinks.push_back(console_sink);
  }
  if (!file_path.empty()) {
    std::filesystem::path path(file_path);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_path, true);
    file_sink->set_pattern("%Y-%m-%dT%H:%M:%S.%f%z CPP %n %l: %v [%g]:line %# :: %!");
    sinks.push_back(file_sink);
  }
  if (sinks.empty()) {
    core_logger_.reset();
    geo_logger_.reset();
    pos_logger_.reset();
    dyn_logger_.reset();
    return;
  }
  const auto level_enum = ParseLevel(level);
  auto register_stage_logger = [&](const std::string& suffix) -> std::shared_ptr<spdlog::logger> {
    const std::string full_name = suffix.empty() ? logger_name : logger_name + "." + suffix;
    spdlog::drop(full_name);
    auto logger = std::make_shared<spdlog::logger>(full_name, sinks.begin(), sinks.end());
    logger->set_level(level_enum);
    spdlog::register_logger(logger);
    return logger;
  };
  core_logger_ = register_stage_logger("");
  geo_logger_ = register_stage_logger("geo");
  pos_logger_ = register_stage_logger("pos");
  dyn_logger_ = register_stage_logger("dyn");
  if (core_logger_) {
    SPDLOG_LOGGER_INFO(core_logger_, "ScoreCalculator loggers configured (console={}, file='{}')", enable_console, file_path);
  }
}

namespace {

// Signed distance helper for both Algorithm 2 (knife clearance) and Algorithm 3 (lever arm).
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

// Algorithm 2: estimate E_fin via nearest neighbor distance in the subset.
double ScoreCalculator::computeMinPairwiseDistance(const PointCloud& subset) const {
  if (subset.size() < 2) {
    return 0.0;
  }
  pcl::KdTreeFLANN<PointT> local_tree;
  auto subset_ptr = subset.makeShared();
  local_tree.setInputCloud(subset_ptr);
  double min_dist = std::numeric_limits<double>::max();
  std::vector<int> nn_indices(2);
  std::vector<float> nn_dists(2);
  for (std::size_t i = 0; i < subset.size(); ++i) {
    if (local_tree.nearestKSearch(subset_ptr->points[i], 2, nn_indices, nn_dists) >= 2) {
      const double dist = std::sqrt(static_cast<double>(nn_dists[1]));
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

Eigen::Vector3d ScoreCalculator::computeCentroid(const PointCloud& subset) const {
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  if (subset.empty()) {
    return centroid;
  }
  for (const auto& point : subset) {
    centroid.x() += static_cast<double>(point.x);
    centroid.y() += static_cast<double>(point.y);
    centroid.z() += static_cast<double>(point.z);
  }
  centroid /= static_cast<double>(subset.size());
  return centroid;
}

// Algorithm 4: construct grasp matrix G = [I; skew(p_i)] for each contact.
Eigen::MatrixXd ScoreCalculator::buildGraspMatrix(const Eigen::VectorXi& indices,
                                                  const Eigen::Vector3d& center,
                                                  bool planar_constraint) const {
  const Eigen::Index contacts = indices.size();
  Eigen::MatrixXd G(6, 3 * contacts);
  G.setZero();
  for (Eigen::Index j = 0; j < contacts; ++j) {
    const int idx = indices(j);
    if (idx < 0 || idx >= static_cast<int>(cloud_->size())) {
      continue;
    }
    const auto& p = cloud_->points[static_cast<std::size_t>(idx)];
    Eigen::Vector3d point(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
    point -= center;
    Eigen::Matrix3d skew;
    skew << 0, -point.z(), point.y(), point.z(), 0, -point.x(), -point.y(), point.x(), 0;
    G.block<3, 3>(0, 3 * j) = Eigen::Matrix3d::Identity();
    G.block<3, 3>(3, 3 * j) = skew;
  }
  if (planar_constraint) {
    return ReducePlanarGraspMatrix(G);
  }
  return G;
}

double ScoreCalculator::calcForceResidual(const Eigen::VectorXi& indices,
                                          const Eigen::VectorXd& wrench,
                                          const Eigen::Vector3d& center,
                                          bool planar_constraint,
                                          const Eigen::VectorXd& f) const {
  if (indices.size() == 0) {
    return std::numeric_limits<double>::infinity();
  }
  if (f.size() != indices.size() * 3) {
    return std::numeric_limits<double>::infinity();
  }
  Eigen::MatrixXd G = buildGraspMatrix(indices, center, planar_constraint);
  Eigen::VectorXd wrench_used = wrench;
  if (planar_constraint) {
    wrench_used = ReducePlanarWrench(wrench);
  }
  Eigen::VectorXd residual_vec = G * f + wrench_used;
  Eigen::VectorXd residual_full = residual_vec;
  if (planar_constraint) {
    residual_full = ExpandPlanarResidual(residual_vec);
  }
  if (dyn_logger_) {
    SPDLOG_LOGGER_INFO(dyn_logger_,
                       "Force residual vec=[{:+03.4f}, {:+03.4f}, {:+03.4f}, {:+03.4f}, {:+03.4f}, {:+03.4f}]",
                       residual_full(0),
                       residual_full(1),
                       residual_full(2),
                       residual_full(3),
                       residual_full(4),
                       residual_full(5));
  }
  return residual_vec.norm();
}

bool ScoreCalculator::checkRandomForceBalance(const Eigen::VectorXi& indices,
                                              const Eigen::VectorXd& wrench,
                                              const Eigen::Vector3d& center,
                                              bool planar_constraint,
                                              double balance_threshold,
                                              const Eigen::VectorXd& f_init_input) const {
  if (!cloud_ || cloud_->empty() || indices.size() == 0) {
    return false;
  }
  if (wrench.size() != 6 || balance_threshold < 0.0) {
    return false;
  }
  for (Eigen::Index j = 0; j < indices.size(); ++j) {
    const int idx = indices(j);
    if (idx < 0 || idx >= static_cast<int>(cloud_->size())) {
      return false;
    }
  }

  Eigen::MatrixXd G = buildGraspMatrix(indices, center, planar_constraint);
  Eigen::VectorXd f_init(3 * indices.size());
  const bool use_random = (f_init_input.size() == 1 && std::isnan(f_init_input(0)));
  if (use_random) {
    f_init.setRandom();
  } else {
    if (f_init_input.size() != f_init.size()) {
      return false;
    }
    f_init = f_init_input;
  }

  // Print out all information for debugging.
  std::cout << "Grasp matrix G:\n" << G << std::endl;
  std::cout << "Initial force f_init:\n" << f_init.transpose() << std::endl;
  std::cout << "Wrench:\n" << wrench.transpose() << std::endl;

  Eigen::MatrixXd G_pinv = PseudoInverse(G, 1e-9);
  Eigen::VectorXd wrench_used = wrench;
  if (planar_constraint) {
    wrench_used = ReducePlanarWrench(wrench);
  }

  std::cout << "Pseudo-inverse of G:\n" << G_pinv << std::endl;
  std::cout << "Wrench used:\n" << wrench_used.transpose() << std::endl;

  Eigen::VectorXd t = (-wrench_used) - G * f_init;
  Eigen::VectorXd f = f_init + G_pinv * t;

  std::cout << "Force Error t:\n" << t.transpose() << std::endl;
  std::cout << "Corrected force f:\n" << f.transpose() << std::endl;

  Eigen::VectorXd residual_vec = G * f + wrench_used;
  std::cout << "Residual vector:\n" << residual_vec.transpose() << std::endl;
  Eigen::VectorXd residual_full = residual_vec;
  if (planar_constraint) {
    residual_full = ExpandPlanarResidual(residual_vec);
  }
  double residual = residual_vec.norm();
  if (dyn_logger_) {
    SPDLOG_LOGGER_INFO(dyn_logger_, "Random force balance residual={:.6f}", residual);
  }
  if (dyn_logger_) {
    SPDLOG_LOGGER_INFO(
        dyn_logger_,
        "Random force balance residual vec=[{:+03.4f}, {:+03.4f}, {:+03.4f}, {:+03.4f}, {:+03.4f}, {:+03.4f}]",
        residual_full(0),
        residual_full(1),
        residual_full(2),
        residual_full(3),
        residual_full(4),
        residual_full(5));
  }
  return residual <= balance_threshold;
}

ScoreCalculator::CandidateMatrix ScoreCalculator::filterByGeoScore(
    const Eigen::Ref<const CandidateMatrix>& candidate_indices,
    const Eigen::Vector3d& knife_p,
    const Eigen::Vector3d& knife_n,
    double table_z) const {
  if (candidate_indices.rows() == 0 || candidate_indices.cols() == 0 || !cloud_ || cloud_->empty()) {
    last_geo_order_.resize(0);
    return CandidateMatrix(0, candidate_indices.cols());
  }

  if (geo_logger_) {
    SPDLOG_LOGGER_DEBUG(geo_logger_, "GeoFilter invoked with {} rows", candidate_indices.rows());
  }
  std::vector<RowScore> row_scores;
  row_scores.reserve(static_cast<std::size_t>(candidate_indices.rows()));

  PointCloud subset;
  subset.reserve(static_cast<std::size_t>(candidate_indices.cols()));

  for (Eigen::Index row = 0; row < candidate_indices.rows(); ++row) {
    bool row_valid = true;
    subset.clear();
    for (Eigen::Index col = 0; col < candidate_indices.cols(); ++col) {
      const int idx = candidate_indices(row, col);
      if (idx < 0 || idx >= static_cast<int>(cloud_->size())) {
        row_valid = false;
        break;
      }
      subset.push_back(cloud_->points[static_cast<std::size_t>(idx)]);
    }
    if (!row_valid) {
      continue;
    }
    subset.width = static_cast<uint32_t>(subset.size());
    subset.height = 1;
    subset.is_dense = true;
    Eigen::Vector3d centroid = computeCentroid(subset);
    const double e_fin = computeMinPairwiseDistance(subset);
    const double e_knf = distanceToPlane(centroid, knife_p, knife_n);
    double min_table = std::numeric_limits<double>::max();
    for (const auto& point : subset) {
      const double dist = static_cast<double>(point.z) - table_z;
      if (dist < min_table) {
        min_table = dist;
      }
    }
    const double e_tbl = std::max(min_table, 0.0);
    row_scores.push_back({row, e_fin, e_knf, e_tbl});
  }

  if (row_scores.empty()) {
    last_geo_order_.resize(0);
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

  last_geo_order_.resize(static_cast<Eigen::Index>(order.size()));
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(order.size()); ++i) {
    last_geo_order_(i) = order[static_cast<std::size_t>(i)].second;
  }

  Eigen::Index keep = static_cast<Eigen::Index>(order.size());
  if (geo_ratio_ > 0.0 && geo_ratio_ < 1.0) {
    keep = std::max<Eigen::Index>(1, static_cast<Eigen::Index>(std::round(geo_ratio_ * keep)));
  }
  Eigen::Index keep_ratio = keep;
  std::vector<Eigen::Index> sampled;
  sampled.reserve(static_cast<std::size_t>(keep_ratio));
  for (Eigen::Index i = 0; i < keep_ratio; ++i) {
    sampled.push_back(i);
  }
  if (max_candidates_ > 0 && max_candidates_ < keep_ratio) {
    std::mt19937 rng(geo_seed_);
    std::shuffle(sampled.begin(), sampled.end(), rng);
    sampled.resize(static_cast<std::size_t>(max_candidates_));
  }

  CandidateMatrix output(static_cast<Eigen::Index>(sampled.size()), candidate_indices.cols());
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(sampled.size()); ++i) {
    output.row(i) = candidate_indices.row(order[static_cast<std::size_t>(sampled[static_cast<std::size_t>(i)])].second);
  }
  if (geo_logger_) {
    SPDLOG_LOGGER_INFO(geo_logger_, "GeoFilter kept {} rows out of {}", output.rows(), row_scores.size());
  }
  return output;
}

Eigen::VectorXi ScoreCalculator::lastGeoOrder() const {
  return last_geo_order_;
}

// Algorithm 3: PCA-based directional alignment.
Eigen::VectorXd ScoreCalculator::calcPositionalScores(
    const Eigen::Ref<const CandidateMatrix>& candidate_indices,
    const Eigen::Vector3d& knife_p,
    const Eigen::Vector3d& knife_n) const {
  if (candidate_indices.rows() == 0 || candidate_indices.cols() == 0 || !cloud_ || cloud_->empty()) {
    return Eigen::VectorXd(0);
  }
  if (pos_logger_) {
    SPDLOG_LOGGER_DEBUG(pos_logger_, "Positional score on {} rows", candidate_indices.rows());
  }
  Eigen::VectorXd scores(candidate_indices.rows());
  PointCloud subset;
  subset.reserve(static_cast<std::size_t>(candidate_indices.cols()));
  Eigen::Vector3d normalized_kn = knife_n.normalized();
  if (!std::isfinite(normalized_kn.norm()) || normalized_kn.norm() < 1e-6) {
    normalized_kn = Eigen::Vector3d(0.0, 0.0, 1.0);
  }

  for (Eigen::Index row = 0; row < candidate_indices.rows(); ++row) {
    subset.clear();
    bool row_valid = true;
    for (Eigen::Index col = 0; col < candidate_indices.cols(); ++col) {
      const int idx = candidate_indices(row, col);
      if (idx < 0 || idx >= static_cast<int>(cloud_->size())) {
        row_valid = false;
        break;
      }
      subset.push_back(cloud_->points[static_cast<std::size_t>(idx)]);
    }
    double score = 0.0;
    if (row_valid && subset.size() >= 2) {
      subset.width = static_cast<uint32_t>(subset.size());
      subset.height = 1;
      subset.is_dense = true;
      Eigen::Vector3d centroid = computeCentroid(subset);
      Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
      for (const auto& point : subset) {
        Eigen::Vector3d diff(static_cast<double>(point.x) - centroid.x(),
                             static_cast<double>(point.y) - centroid.y(),
                             static_cast<double>(point.z) - centroid.z());
        cov += diff * diff.transpose();
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
      if (solver.info() == Eigen::Success) {
        Eigen::Vector3d principal = solver.eigenvectors().col(2);
        if (principal.norm() > 1e-6) {
          principal.normalize();
          score = 1.0 - std::abs(principal.dot(normalized_kn));
        }
      }
    }
    scores(row) = std::clamp(score, 0.0, 1.0);
  }
  return scores;
}

// Algorithm 3: distance from candidate centroid to knife plane.
Eigen::VectorXd ScoreCalculator::calcPositionalDistances(
    const Eigen::Ref<const CandidateMatrix>& candidate_indices,
    const Eigen::Vector3d& knife_p,
    const Eigen::Vector3d& knife_n) const {
  if (candidate_indices.rows() == 0 || candidate_indices.cols() == 0 || !cloud_ || cloud_->empty()) {
    return Eigen::VectorXd(0);
  }
  if (pos_logger_) {
    SPDLOG_LOGGER_DEBUG(pos_logger_, "Positional distance on {} rows", candidate_indices.rows());
  }
  Eigen::VectorXd scores(candidate_indices.rows());
  PointCloud subset;
  subset.reserve(static_cast<std::size_t>(candidate_indices.cols()));
  Eigen::Vector3d normalized_kn = knife_n.normalized();
  if (!std::isfinite(normalized_kn.norm()) || normalized_kn.norm() < 1e-6) {
    normalized_kn = Eigen::Vector3d(0.0, 0.0, 1.0);
  }
  for (Eigen::Index row = 0; row < candidate_indices.rows(); ++row) {
    subset.clear();
    bool row_valid = true;
    for (Eigen::Index col = 0; col < candidate_indices.cols(); ++col) {
        const int idx = candidate_indices(row, col);
        if (idx < 0 || idx >= static_cast<int>(cloud_->size())) {
            row_valid = false;
            break;
        }
        subset.push_back(cloud_->points[static_cast<std::size_t>(idx)]);
    }
    double score = 0.0;
    if (row_valid && subset.size() >= 1) {
        Eigen::Vector3d centroid = computeCentroid(subset);
        score = distanceToPlane(centroid, knife_p, normalized_kn);
    }
    scores(row) = score;
  }
  const double min_v = scores.minCoeff();
  const double max_v = scores.maxCoeff();
  const double range = max_v - min_v;
  if (std::isfinite(range) && range >= 1e-9) {
    scores = (scores.array() - min_v) / range;
  } else {
    scores.setZero();
  }
  return scores;
}

// Algorithm 4: evaluate dynamics feasibility, residuals, and score terms.
// 算法4：评估动力学可行性、残差和评分项
Eigen::VectorXd ScoreCalculator::calcDynamicsScores(
    const Eigen::Ref<const CandidateMatrix>& candidate_indices,
    const Eigen::VectorXd& wrench,
    const Eigen::Vector3d& center,
    bool planar_constraint,
    double friction_coef,
    double friction_angle_deg,
    int max_attempts,
    double balance_threshold,
    double force_min,
    double force_max,
    double cone_angle_max_deg) const {
  // 验证输入：检查候选索引矩阵和点云是否有效
  if (candidate_indices.rows() == 0 || candidate_indices.cols() == 0 || !cloud_ || cloud_->empty()) {
    return Eigen::VectorXd(0);
  }
  if (dyn_logger_) {
    SPDLOG_LOGGER_INFO(
        dyn_logger_,
        "Dynamics scoring rows={} wrench_norm={:.4f} attempts={} balance={:.6f} planar={} force=[{:.3f},{:.3f}] cone_max={:.2f}deg",
        candidate_indices.rows(),
        wrench.norm(),
        max_attempts,
        balance_threshold,
        planar_constraint,
        force_min,
        force_max,
        cone_angle_max_deg);
  }
  
  // 初始化评分向量，每行候选配置对应一个评分
  const Eigen::Index rows = candidate_indices.rows();
  Eigen::VectorXd scores(rows);
  last_dyn_attempts_.clear();
  last_dyn_attempts_.resize(static_cast<std::size_t>(rows));
  
  // 将摩擦角从度转换为弧度，并预计算正切值用于摩擦锥约束检查
  double friction_angle_rad = friction_angle_deg * M_PI / 180.0;
  double tan_angle = std::tan(friction_angle_rad);
  if (!std::isfinite(tan_angle) || tan_angle < 0.0) {
    tan_angle = 0.0;
  }
  if (max_attempts <= 0) {
    max_attempts = 1;
  }
  (void)friction_coef;
  if (force_min <= 0.0) {
    force_min = 0.001;
  }
  if (force_max <= force_min) {
    force_max = force_min + 1.0;
  }
  if (cone_angle_max_deg < 0.0) {
    cone_angle_max_deg = 0.0;
  }
  double cone_angle_max_rad = std::clamp(cone_angle_max_deg, 0.0, 89.0) * M_PI / 180.0;
  SPDLOG_LOGGER_INFO(dyn_logger_, "Friction angle: {:.4f} deg, {:.4f} rad", cone_angle_max_deg, cone_angle_max_rad);

  std::mt19937 rng(dyn_seed_);
  std::uniform_real_distribution<double> angle_dist(0.0, cone_angle_max_rad);
  std::uniform_real_distribution<double> normal_dist(force_min, force_max);
  
  // Visualize point cloud with normals using PCL
    // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud with Normals"));
    // viewer->setBackgroundColor(0.3, 0.3, 0.3);
    // pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_color_handler(cloud_, 255, 255, 255);
    // viewer->addPointCloud<PointT>(cloud_, cloud_color_handler, "cloud");
    // SPDLOG_LOGGER_INFO(dyn_logger_, "Point cloud size: {}", cloud_->size());
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
    // viewer->addPointCloudNormals<PointT, PointT>(cloud_, cloud_, 1, 0.02, "normals");
    // viewer->addCoordinateSystem(0.1);
    // viewer->initCameraParameters();
    
    // if (dyn_logger_) {
    //   SPDLOG_LOGGER_INFO(dyn_logger_, "PCL Visualizer created. Press 'q' to close and continue.");
    // }
    
    // viewer->spin();
    // viewer->close();

  std::cout << "Center: [" << center.transpose() << "]" << std::endl;

  Eigen::MatrixX3d RawScores(rows, 3);
  RawScores.setZero();
  // 遍历每个候选抓取配置（每行代表一组接触点）
  for (Eigen::Index i = 0; i < rows; ++i) {
    if (dyn_logger_ && (i % 50 == 0 || i + 1 == rows)) {
      SPDLOG_LOGGER_INFO(dyn_logger_, "Dynamics progress {}/{}", i + 1, rows);
    }
    Eigen::VectorXi indices = candidate_indices.row(i);
    
    // 验证当前行的所有索引是否在点云范围内
    bool valid = true;
    for (Eigen::Index j = 0; j < indices.size(); ++j) {
      if (indices(j) < 0 || indices(j) >= static_cast<int>(cloud_->size())) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      scores(i) = 0.0;
      continue;
    }
    
    // 构建抓取矩阵 G (6×3n)，其中 n 是接触点数量
    // G = [I, I, ...; skew(p1), skew(p2), ...] 用于映射接触力到物体扭矩
    Eigen::MatrixXd G = buildGraspMatrix(indices, center, planar_constraint);
    Eigen::MatrixXd G_pinv = PseudoInverse(G, 1e-9);
    Eigen::VectorXd wrench_used = wrench;
    if (planar_constraint) {
      wrench_used = ReducePlanarWrench(wrench);
    }

    Eigen::Index contact_count = indices.size();
    double best_score = -std::numeric_limits<double>::infinity();
    bool has_valid = false;
    auto& attempts = last_dyn_attempts_[static_cast<std::size_t>(i)];
    attempts.clear();
    attempts.reserve(static_cast<std::size_t>(max_attempts));

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
      Eigen::VectorXd f_init(3 * contact_count);

        for (Eigen::Index j = 0; j < contact_count; ++j) {
          const auto& p = cloud_->points[static_cast<std::size_t>(indices(j))];
          Eigen::Vector3d normal = SafeNormal(p);
          Eigen::Vector3d sample = SampleForceInCone(normal, rng, angle_dist, normal_dist);
          f_init.segment(3 * j, 3) = sample;
          
          double force_angle = std::acos(normal.dot(sample) / (normal.norm() * sample.norm())) * 180.0 / M_PI;
          // SPDLOG_LOGGER_INFO(dyn_logger_, "Sampled force at contact {}: [{:+03.4f}, {:+03.4f}, {:+03.4f}]  normal [{:+03.4f}, {:+03.4f}, {:+03.4f}], point [{:+03.4f}, {:+03.4f}, {:+03.4f}], angle {:.4f} deg", j, 
          //   sample(0), sample(1), sample(2), normal(0), normal(1), normal(2), p.x, p.y, p.z, force_angle);
        assert(force_angle <= cone_angle_max_deg + 1e-2);
      }


      // char a;
      // std::cin >> a;
      

      // t = (-wrench) - G * f_init
      Eigen::VectorXd t = (-wrench_used) - G * f_init;
      Eigen::VectorXd f = f_init + G_pinv * t;

      
      for (Eigen::Index j = 0; j < contact_count; ++j) {
        const auto& p = cloud_->points[static_cast<std::size_t>(indices(j))];
        Eigen::Vector3d normal = SafeNormal(p);
        Eigen::Vector3d sample = f.segment(3 * j, 3);
        double force_angle = std::acos(normal.dot(sample) / (normal.norm() * sample.norm())) * 180.0 / M_PI;
      }
      
      Eigen::VectorXd residual_vec = G * f + wrench_used;

      // std::cout << "=============================================" << std::endl;  // Print segments
      // std::cout << "----------------------------" << std::endl;  // Print segments
      // std::cout << "Grasp matrix G:\n" << G << std::endl;
      // std::cout << "Initial force f_init:\n" << f_init.transpose() << std::endl;
      // std::cout << "Wrench:\n" << wrench.transpose() << std::endl;
      // std::cout << "Pseudo-inverse of G:\n" << G_pinv << std::endl;
      // std::cout << "Wrench used:\n" << wrench_used.transpose() << std::endl;
      // std::cout << "Force Error t:\n" << t.transpose() << std::endl;
      // std::cout << "Corrected force f:\n" << f.transpose() << std::endl;
      // std::cout << "Residual vector:\n" << residual_vec.transpose() << std::endl;
      // std::cout << "----------------------------" << std::endl;  // Print segments
      // std::cout << "=============================================" << std::endl;  // Print segments
      // std::cout << std::endl;  // Print segments
      // std::cout << std::endl;  // Print segments

      double residual = residual_vec.norm();
      // std::cout << "residual_vec: " << residual_vec.transpose() << "wrench: " << wrench_used.transpose() << std::endl;
      bool balance_ok = residual <= balance_threshold;
      // if (balance_ok && dyn_logger_) {
      //   SPDLOG_LOGGER_INFO(dyn_logger_, "-=-=-=-= Attempt [{},{}] succeeded: residual={:.6f}", i, attempt + 1, residual);
      // }

      bool cone_ok = true;
      std::vector<double> magnitudes;
      magnitudes.reserve(static_cast<std::size_t>(contact_count));
      double e_dir = 0.0;
      double sum_mag = 0.0;

      for (Eigen::Index j = 0; j < contact_count; ++j) {
        const auto& p = cloud_->points[static_cast<std::size_t>(indices(j))];
        Eigen::Vector3d normal = SafeNormal(p);
        Eigen::Vector3d contact_force = f.segment(3 * j, 3);
        double mag = contact_force.norm();
        magnitudes.push_back(mag);
        sum_mag += mag;
        if (mag > 1e-9) {
          e_dir += normal.dot(contact_force / mag);
        }

        // double normal_component = contact_force.dot(normal);
        // Eigen::Vector3d tangential_vec = contact_force - normal_component * normal;
        // double tangential = tangential_vec.norm();
        // if (normal_component <= 0.0 || tangential > normal_component * tan_angle + 1e-9) {
        //   cone_ok = false;
        // }

        auto f_dir = contact_force.normalized();
        const double nf_angle = std::acos(normal.dot(f_dir)) * 180.0 / M_PI;
        if (nf_angle > cone_angle_max_deg + 1e-2) {
          cone_ok = false;
        }
      }

      double e_mag = -sum_mag;
      double e_var = 0.0;
      if (magnitudes.size() > 1) {
        double mean = sum_mag / magnitudes.size();
        double var = 0.0;
        for (double m : magnitudes) {
          var += (m - mean) * (m - mean);
        }
        var /= magnitudes.size();
        e_var = -var;
      }

      double total = e_mag + e_dir + e_var;
      attempts.emplace_back(f, f_init, e_mag, e_dir, e_var, balance_ok && cone_ok);

      if (balance_ok && cone_ok) {
        has_valid = true;
        // if (dyn_logger_) {
        //   SPDLOG_LOGGER_INFO(dyn_logger_,
        //                      "  Valid forces found: e_mag={:.6f} e_dir={:.6f} e_var={:.6f} total={:.6f}",
        //                      e_mag,
        //                      e_dir,
        //                      e_var,
        //                      total);
        // }
        if (total > best_score) {
          best_score = total;
          RawScores(i, 0) = e_mag;
          RawScores(i, 1) = e_dir;
          RawScores(i, 2) = e_var;
        }
      }
      else {
        // if (dyn_logger_) {
        //   SPDLOG_LOGGER_INFO(dyn_logger_, "Attempt [{},{}] failed: balance_ok={} cone_ok={} residual={:.6f}", i, attempt + 1, balance_ok, cone_ok, residual);
        // }
      }
    }
    // if (dyn_logger_) {
    //   SPDLOG_LOGGER_INFO(dyn_logger_, "Found [{}] attempts for candidate {}", attempts.size(), i);
    // }
    // std::sort(attempts.begin(), attempts.end(), [](const ForceAttempt& a, const ForceAttempt& b) {
    //   double total_a = std::get<2>(a) + std::get<3>(a) + std::get<4>(a);
    //   double total_b = std::get<2>(b) + std::get<3>(b) + std::get<4>(b);
    //   return total_a > total_b;
    // });

    // scores(i) = has_valid ? best_score : -std::numeric_limits<double>::infinity();
  }
  // Normalize RawScores over all $P$.
  for (int col = 0; col < RawScores.cols(); ++col) {
    // auto& ColumnRef = RawScores.col(col);
    const double* pWeight = &(this->force_weights_.w_mag);
    pWeight += col;

    const double min_v = RawScores.col(col).array().isFinite().select(RawScores.col(col).array(), std::numeric_limits<double>::infinity()).minCoeff();
    const double max_v = RawScores.col(col).array().isFinite().select(RawScores.col(col).array(), -std::numeric_limits<double>::infinity()).maxCoeff();
    const double range = max_v - min_v;
    if (std::isfinite(range) && range >= 1e-9) {
      RawScores.col(col).array() = (RawScores.col(col).array() - min_v) / range;
    } else {
      if (range < 1e-9)
      {
        if (max_v < std::numeric_limits<double>::infinity() && min_v > -std::numeric_limits<double>::infinity())
        {
          RawScores.col(col).setOnes();
        }
      }
      else {
        RawScores.col(col).setZero();
      }
    }
    RawScores.col(col) *= *pWeight;

    if (dyn_logger_) {
      SPDLOG_LOGGER_INFO(dyn_logger_, " Dynamics score column {} normalized with weight {:.4f}; max {:.6f} min {:.6f}", col, *pWeight, max_v, min_v);
    }
  }

  scores = RawScores.rowwise().sum();

  
  if (dyn_logger_) {
    SPDLOG_LOGGER_DEBUG(dyn_logger_, "Dynamics scoring complete");
  }
  return scores;
}
