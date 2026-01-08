#include "ScoreCalculator.h"

#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace {

spdlog::level::level_enum ParseLevel(const std::string& level) {
  try {
    return spdlog::level::from_str(level);
  } catch (const spdlog::spdlog_ex&) {
    return spdlog::level::info;
  }
}

}  // namespace

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
Eigen::MatrixXd ScoreCalculator::buildGraspMatrix(const Eigen::VectorXi& indices) const {
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
    Eigen::Matrix3d skew;
    skew << 0, -point.z(), point.y(), point.z(), 0, -point.x(), -point.y(), point.x(), 0;
    G.block<3, 3>(0, 3 * j) = Eigen::Matrix3d::Identity();
    G.block<3, 3>(3, 3 * j) = skew;
  }
  return G;
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
  if (max_candidates_ > 0 && max_candidates_ < keep) {
    keep = static_cast<Eigen::Index>(max_candidates_);
  }

  CandidateMatrix output(keep, candidate_indices.cols());
  for (Eigen::Index i = 0; i < keep; ++i) {
    output.row(i) = candidate_indices.row(order[static_cast<std::size_t>(i)].second);
  }
  if (geo_logger_) {
    SPDLOG_LOGGER_INFO(geo_logger_, "GeoFilter kept {} rows out of {}", keep, row_scores.size());
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
    double friction_coef,
    double friction_angle_deg) const {
  // 验证输入：检查候选索引矩阵和点云是否有效
  if (candidate_indices.rows() == 0 || candidate_indices.cols() == 0 || !cloud_ || cloud_->empty()) {
    return Eigen::VectorXd(0);
  }
  if (dyn_logger_) {
    SPDLOG_LOGGER_INFO(dyn_logger_, "Dynamics scoring on {} rows; wrench norm={}", candidate_indices.rows(), wrench.norm());
  }
  
  // 初始化评分向量，每行候选配置对应一个评分
  const Eigen::Index rows = candidate_indices.rows();
  Eigen::VectorXd scores(rows);
  
  // 将摩擦角从度转换为弧度，并预计算正切值用于摩擦锥约束检查
  double friction_angle_rad = friction_angle_deg * M_PI / 180.0;
  double tan_angle = std::tan(friction_angle_rad);
  
  // 遍历每个候选抓取配置（每行代表一组接触点）
  for (Eigen::Index i = 0; i < rows; ++i) {
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
    Eigen::MatrixXd G = buildGraspMatrix(indices);
    
    // 求解最小二乘问题 G*f = -wrench，得到每个接触点的力向量
    Eigen::VectorXd f = G.completeOrthogonalDecomposition().solve(-wrench);
    
    // 计算残差向量 G*f + wrench，衡量抓取平衡性
    Eigen::VectorXd residual_vec = G * f + wrench;
    double residual = residual_vec.norm();
    
    // 平衡度评分：残差越小，平衡度越高
    double balance = 1.0 / (1.0 + residual);
    
    // 初始化接触点分析所需的变量
    Eigen::Index contact_count = indices.size();
    int feasible_contacts = 0;  // 满足摩擦锥约束的接触点数量
    std::vector<double> magnitudes;  // 各接触点的力大小
    magnitudes.reserve(static_cast<std::size_t>(contact_count));
    Eigen::Vector3d net_force = Eigen::Vector3d::Zero();  // 合力
    
    // 遍历每个接触点，检查摩擦锥约束和力的特性
    for (Eigen::Index j = 0; j < contact_count; ++j) {
      const auto& p = cloud_->points[static_cast<std::size_t>(indices(j))];
      
      // 获取接触点的法向量
      Eigen::Vector3d normal(static_cast<double>(p.normal_x), static_cast<double>(p.normal_y), static_cast<double>(p.normal_z));
      if (normal.norm() < 1e-6) {
        // 如果法向量无效，使用默认的竖直方向
        normal = Eigen::Vector3d(0.0, 1.0, 0.0);
      } else {
        normal.normalize();
      }
      
      // 提取该接触点的三维力向量
      Eigen::Vector3d contact_force = f.segment(3 * j, 3);
      
      // 累加合力和记录力的大小
      net_force += contact_force;
      magnitudes.push_back(contact_force.norm());
      
      // 将接触力分解为法向分量和切向分量
      double normal_component = contact_force.dot(normal);
      Eigen::Vector3d tangential_vec = contact_force - normal_component * normal;
      double tangential = tangential_vec.norm();
      
      // 计算摩擦锥约束：切向力 ≤ μ * |法向力| * tan(α)
      double limit = friction_coef * std::max(normal_component, 0.0) * tan_angle;
      
      // 检查是否满足约束：法向力为正且切向力在限制内
      if (normal_component > 0.0 && tangential <= limit + 1e-9) {
        ++feasible_contacts;
      }
    }
    
    // 可行性评分：满足摩擦锥约束的接触点占比
    double feasibility = contact_count > 0 ? static_cast<double>(feasible_contacts) / static_cast<double>(contact_count) : 0.0;
    
    // 计算力的大小匹配度：合力大小与期望外力（wrench）的匹配程度
    double wrench_force = wrench.head(3).norm();  // 期望外力的大小
    double net_force_norm = net_force.norm();     // 实际合力的大小
    double e_mag = 1.0 / (1.0 + std::abs(net_force_norm - wrench_force));
    
    // 计算力的方向匹配度：合力方向与期望方向的一致性
    double e_dir = 0.0;
    if (net_force_norm > 1e-9 && wrench_force > 1e-9) {
      // 余弦相似度：合力方向与负wrench方向的点积
      e_dir = std::max(0.0, net_force.normalized().dot((-wrench.head(3)).normalized()));
    }
    
    // 计算力的均匀性：各接触点力大小的一致性（方差越小越好）
    double e_var = 0.0;
    if (magnitudes.size() > 1) {
      // 计算力大小的均值
      double mean = std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0) / magnitudes.size();
      // 计算方差
      double var = 0.0;
      for (double m : magnitudes) {
        var += (m - mean) * (m - mean);
      }
      var /= magnitudes.size();
      // 方差越小，均匀性评分越高
      e_var = 1.0 / (1.0 + std::sqrt(var));
    } else if (!magnitudes.empty()) {
      // 单个接触点时均匀性为满分
      e_var = 1.0;
    }
    
    // 综合评分：五项指标的平均值
    // feasibility: 摩擦锥可行性, balance: 力平衡性, e_mag: 力大小匹配度
    // e_dir: 力方向匹配度, e_var: 力均匀性
    double combined = (feasibility + balance + e_mag + e_dir + e_var) / 5.0;
    scores(i) = std::clamp(combined, 0.0, 1.0);
  }
  
  if (dyn_logger_) {
    SPDLOG_LOGGER_DEBUG(dyn_logger_, "Dynamics scoring complete");
  }
  return scores;
}
