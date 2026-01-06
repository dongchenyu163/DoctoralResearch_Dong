# 需求文档：机器人食品切割任务中的指尖把持点搜索系统规格书（Rev2）

> 目标：复现并实装论文《Searching Method for Holding Points during Robotic Food Cutting》中的 holding point set 搜索算法，并以工程化方式保证：可复现、可调参、可测时、可调试、可扩展。

---

## 0. 总览

### 0.1 核心目标

- **输入**：食品点云（Point Cloud / Mesh）、包丁轨迹（pose + velocity 序列）、物理参数（摩擦系数、断裂韧性、压力分布等）、权重参数、系统超参数。
- **输出**：最优的 \(N\) 指把持点集合（Holding Point Set）及其对应的一组指尖力解（可选输出：每个 time step 的中间评分 / 诊断信息）。
- **核心功能**：
  - 几何约束下的候选集筛选（低成本）
  - 位置评分（positional score）
  - 刀具 wrench 计算（fracture + friction）
  - 在摩擦锥约束下生成可行 finger forces，并据此计算动力学评分（dynamic score）
  - 沿整个轨迹累积分数，输出总分最高的 holding point set

### 0.2 关键设计原则（Algorithmic Invariants）

为避免复现偏差与后期 debug 困难，系统必须显式遵守以下不变式：

1. \(\Omega_{low}\)（低密度点云）在整个 knife trajectory 生命周期内**只读且不变**：
   - 点数 \(M\) 不变
   - index 语义不变（0..M-1）
   - 坐标必须在同一 world frame 下保持一致
2. \(\mathbb{P}_{all\_indices}\)（所有 \(C_M^N\) 组合）**只生成一次**，后续仅通过索引筛除。
3. 分数累积字典 \(\mathbb{S}\) 的语义：
   - 每个 time step：对当前候选集计算并**累加** \(S_{pos}+S_{force}\)
   - 一旦某组合被标记为 \(-\infty\)（等价于 Algorithm 1 的 lines 10–12），该组合在后续 time step **永不恢复**。
4. 归一化（min-max）仅用于“当前 time step 的候选集相对排序”，**不改变**跨时间累积语义。

---

## 1. 系统架构

采用 **Python + C++** 混合架构：

- **Python**：顶层控制、数据 IO/管理、mesh boolean、可视化、轨迹循环、日志与测时汇总。
- **C++**：KD-Tree / 法向计算、几何与位置评分、动力学评分核心（OpenMP）、高性能数值计算（Eigen）。

### 1.1 技术栈

- Python 3.10+
  - open3d：点云读取/预处理/可视化
  - trimesh：网格处理与布尔运算（可选依赖后端）
  - numpy：矩阵运算与数据传递
  - pybind11：Python ↔ C++ 绑定
  - logging：日志
- C++ / CMake
  - Eigen 3：矩阵/向量
  - PCL 1.12：点云处理（KD-Tree、法向估计、MLS等，可按需）
  - OpenMP：并行

### 1.2 模块划分（高内聚低耦合）

- Config Module（Python）：解析 JSON 配置，构造强类型配置对象。
- Geometry Engine（Python/C++）：
  - Python：刀具 mesh 生成、mesh boolean、接触面提取
  - C++：KD-Tree、法向、距离/投影、几何评分
- Physics Engine（Python/C++）：
  - Python：接触面几何数据构造（ContactSurfaceData）
  - C++：fracture / friction 积分与 wrench 输出（可选：全部移到 C++）
- Search Core（Python Control + C++ Worker）：
  - Python：trajectory loop、mask/valid indices 生成、分数累积、最终输出
  - C++：对候选组合并行计算 geo/pos/dyn scores
- Instrumentation（Python/C++）：细粒度测时与诊断输出（可配置开关）

---

## 2. 数据结构与输入定义

### 2.1 配置文件（config.json）

所有超参数外部化，且支持“测时开关 + 粒度配置”。

```json
{
  "preprocess": {
    "downsample_num": 100,
    "normal_estimation_radius": 0.01
  },
  "weights": {
    "geo_score": {"w_fin": 1.0, "w_knf": 4.4, "w_tbl": 6.0},
    "pos_score": {"w_pdir": 5.0, "w_pdis": 4.0},
    "force_score": {"w_mag": 2.0, "w_dir": 2.0, "w_var": 1.0}
  },
  "knife": {
    "edge_angle_deg": 30.0,
    "height": 0.05
  },
  "physics": {
    "friction_coef": 0.5,
    "fracture_toughness": 400.0,
    "pressure_distribution": 3000.0,
    "planar_constraint": true,
    "friction_cone": {"angle_deg": 40.0, "num_sides": 8},
    "force_balance_threshold": 1e-4
  },
  "search": {
    "geo_filter_ratio": 0.6,
    "force_sample_count": 1000
  },
  "instrumentation": {
    "enable_timing": true,
    "enable_detailed_timing": true,
    "emit_per_timestep_report": false,
    "emit_per_candidate_debug": false,
    "timing_output": {
      "format": "jsonl",
      "path": "logs/timing.jsonl"
    },
    "sections": {
      "python": {
        "io": true,
        "preprocess": true,
        "trajectory_loop": true,
        "valid_indices": true,
        "mesh_boolean": true,
        "contact_surface_purify": true,
        "accumulate_scores": true
      },
      "cpp": {
        "kdtree": true,
        "geo_filter": true,
        "pos_score": true,
        "knife_wrench": true,
        "dyn_score_total": true,
        "force_generate": true,
        "force_check": true,
        "grasp_matrix": true,
        "pinv": true,
        "normalize": true
      }
    }
  }
}
```

> 说明：sections 控制“测时采集点”是否启用。关闭时必须接近零开销（仅一次 if 判断）。

### 2.2 编译期常量与术语

- \(N\) / FINGER\_COUNT：手指数（默认 2）。
- \(M\)：\(\Omega_{low}\) 点数。
- \(\Omega_{high}\)：高密度点云，用于精确 mesh/接触面几何处理。
- \(\Omega_{low}\)：低密度点云，用于候选点搜索与评分（只读不变）。
- \(\Omega_g\)：可把持的点云子集（逻辑集合），实现中以 **valid indices 列表** 表示。
- \(\Omega_{c1}, \Omega_{c2}\)：由切割产生的两侧接触面。

### 2.3 刀具定义（Knife Definition）

- 坐标系：
  - 原点：刀刃根部（heel）
  - X：沿刀刃方向
  - Y：刀具中心平面的法向
  - Z：指向刀背
- 形状：横截面等腰三角形（顶角可配），沿 X 方向延伸。

---

## 3. 算法规格（严格对照论文 Algorithm 1–4）

### 3.1 预处理与候选点生成（PREPAREDATA）

**输入**：原始点云 \(\Omega_{high}\)

**步骤**：

1. 接触面计算使用 \(\Omega_{high}\)（保证 boolean/投影精度）。
2. 降采样得到 \(\Omega_{low}\)（点数由 *config.json* 中 `preprocess.downsample_num` 指定），并估计法向 \(\hat{n}_\Omega(p)\)。
3. 生成全集 \(\mathbb{P}_{all\_indices}\)：对索引 \(0..M-1\) 取所有 \(N\) 组合（\(C_M^N\)）。
4. 每个 time step：
   - 计算 valid indices（\(\Omega_g\) 的实现），无坐标拷贝。
   - 生成 \(\mathbb{P}_{init}\)：仅保留所有指尖索引都在 valid indices 内的组合。

### 3.2 几何评分与筛选（FILTERBYGEOSCORE，Algorithm 2）

**执行端**：C++（OpenMP）

- 评分：
  \(S_{geo} = w_{fin}E_{fin} + w_{knf}E_{knf} + w_{tbl}E_{tbl}\)

- 指间距离惩罚：
  \(E_{fin} = \min_{i\neq j} D(\|p_i-p_j\|)\)

- 刀具避让：
  \(E_{knf} = \min_i Dist(p_i,\ \text{KnifePlane})\)

- 桌面避让：
  \(E_{tbl} = \min_i (p_{i,z})\)

- 数值清洗：归一化前剔除 NaN/Inf 或退化组合（例如距离 0 导致除零）。

- 归一化：min-max（仅当前 time step 的候选集）。

- 筛选：保留 top \(\lfloor rN\rfloor\)（r 来自 config.search.geo\_filter\_ratio）。

> 失败模式定义：几何失败属于“合法淘汰”，等价于 \(S=-\infty\)。

### 3.3 位置评分（CALPOSITIONALSCORE，Algorithm 3）

**执行端**：C++

\(S_{pos} = w_{pdir}E_{pdir} + w_{pdis}E_{pdis}\)

- 方向评分：
  \(E_{pdir} = 1 - |PCA(P)\cdot \hat{n}_k|\)
- 距离评分：
  \(E_{pdis} = -\min(Dist(P,\ \text{KnifePlane}))\)

同样要求：归一化前剔除 NaN/Inf。

### 3.4 刀具 wrench 计算（CALKNIFEFORCE）

为降低耦合与提高可替换性，引入中间抽象：

#### 3.4.1 ContactSurfaceData（推荐）

- Python 负责生成接触面几何数据；C++ 负责积分。

```cpp
struct ContactSurfaceData {
    Eigen::MatrixXd points;   // Kx3
    Eigen::MatrixXd normals;  // Kx3
    Eigen::VectorXd areas;    // K
    int side;                 // +1 / -1
};
```

#### 3.4.2 接触面获取与净化（Python）

- Boolean intersection：\(M_{inter} = food\_mesh \cap knife\_mesh\)
- 通过面法向与刀具侧面法向的夹角阈值剔除外表面 patch，得到纯内部接触面 \(\Omega_{contact}\)
- 连通域分离得到 \(\Omega_{c1},\Omega_{c2}\)

#### 3.4.3 力与力矩（C++ 或 Python→C++）

- fracture force（沿刀刃曲线积分）
- friction force（对 \(\Omega_{c1},\Omega_{c2}\) 面积分，压力均匀）
- 输出：\(\mathbf{t}_k\in\mathbb{R}^6\)
- planar constraint：若启用，仅保留 \(f_x,f_y,\tau_z\) 分量（置零其余分量）

### 3.5 动力学评分（CALDYNAMICSCORE，Algorithm 4）

**执行端**：C++

目标：对每个候选 \(P\)，在摩擦锥约束下生成多组可行 finger forces，取最高分。

- 平衡方程：
  \(G\mathbf{f} = -\mathbf{t}_k\)

- 生成策略（论文提出的高通过率方法）：

  1. 在摩擦锥内随机生成 \(\mathbf{f}_{init}\)
  2. 计算误差 \(\mathbf{t}_{diff}=(-\mathbf{t}_k)-G\mathbf{f}_{init}\)
  3. 修正 \(\mathbf{f}=\mathbf{f}_{init}+G^+\mathbf{t}_{diff}\)
  4. 校验摩擦锥 + 残差阈值

- 评分：
  \(S_{force} = \max_j(w_{mag}E_{mag}+w_{dir}E_{dir}+w_{var}E_{var})\)

- 失败模式定义（必须写死）：

  - 物理失败：无可行力解（全部超摩擦锥或残差超阈值）→ \(S=-\infty\)
  - 数值失败：\(G\) 退化导致伪逆不稳定（共线/共点等）→ \(S=-\infty\)

---

## 4. 接口设计（Python ↔ C++，pybind11）

### 4.1 C++ API

```cpp
constexpr int FINGER_COUNT = 2;
using CandidateMatrix = Eigen::Matrix<int, Eigen::Dynamic, FINGER_COUNT>;

struct TimingSectionFlag {
    bool Enable;
};

struct ScoreDebugInfo {
    Eigen::VectorXd geo_raw;
    Eigen::VectorXd pos_raw;
    Eigen::VectorXd force_raw;
};

class ScoreCalculator {
public:
    void setPointCloud(const Eigen::Ref<const Eigen::MatrixXd>& points,
                       const Eigen::Ref<const Eigen::MatrixXd>& normals,
                       const Eigen::Vector3d& center_of_mass);

    void setConfig(const std::map<std::string, double>& weights,
                   const std::map<std::string, double>& physics_params,
                   const std::map<std::string, bool>& timing_sections);

    CandidateMatrix filterByGeoScore(const std::vector<int>& valid_point_indices,
                                     const Eigen::Vector3d& knife_p,
                                     const Eigen::Vector3d& knife_n,
                                     double table_z);

    Eigen::VectorXd calcPositionalScores(const Eigen::Ref<const CandidateMatrix>& candidate_indices,
                                         const Eigen::Vector3d& knife_p,
                                         const Eigen::Vector3d& knife_n);

    Eigen::VectorXd calcDynamicScores(const Eigen::VectorXd& wrench_6d,
                                      const Eigen::Ref<const CandidateMatrix>& candidate_indices);

    // 可选：用于调试/画图
    ScoreDebugInfo calcScoresWithDebug(const Eigen::VectorXd& wrench_6d,
                                       const Eigen::Ref<const CandidateMatrix>& candidate_indices);
};
```

> 注：timing\_sections 用于从 Python 传入哪些 section 需要计时（细粒度开关）。

### 4.2 Python 主流程（示意）

```python
for t, pose in enumerate(knife_trajectory):
    # Python timing: per-step start

    # 1) valid indices
    valid_indices = get_valid_indices(points_low, pose)

    # 2) C++ geo filter
    candidate_indices = calculator.filterByGeoScore(valid_indices, pose.point, pose.normal, table_z)

    # 3) knife wrench
    wrench = calc_knife_wrench_python_or_cpp(...)

    # 4) scores
    s_pos = calculator.calcPositionalScores(candidate_indices, pose.point, pose.normal)
    s_dyn = calculator.calcDynamicScores(wrench, candidate_indices)

    # 5) accumulate
    accumulate_scores(candidate_indices, s_pos, s_dyn)

# final argmax
```

---

## 5. 细粒度执行时间测量（Instrumentation）

### 5.1 目标与原则

- 目标：测到“论文每个函数、甚至子步骤”的耗时，并能定位瓶颈。
- 原则：
  - 默认关闭时：开销应极低（近似一次 if + RAII 构造消除）。
  - 开启时：支持 **per time step**、**per module**、**可选 per candidate**（极重，不建议默认开）。
  - 输出格式：推荐 JSONL（每行一个 event），便于后处理与可视化。

### 5.2 测时颗粒度清单（建议至少实现这些）

#### 5.2.1 Python 侧（每个 time step）

- IO：点云读取、mesh 读取
- preprocess：downsample、normal estimation
- trajectory\_loop：每 step 总耗时
- valid\_indices：计算 \(\Omega_g\) mask / indices
- mesh\_boolean：intersection
- contact\_surface\_purify：剔除外表面 patch、分离 \(\Omega_{c1},\Omega_{c2}\)
- accumulate\_scores：字典累加 + 最终 argmax

#### 5.2.2 C++ 侧

- kdtree：构建/查询（若每次重建，需单独计时）
- geo\_filter：整体耗时
  - geo\_Efin
  - geo\_Eknf
  - geo\_Etbl
  - geo\_normalize
  - geo\_sort\_select
- pos\_score：整体耗时
  - pos\_PCA
  - pos\_plane\_dist
  - pos\_normalize
- knife\_wrench（若移到 C++）：
  - fracture\_integral
  - friction\_integral
- dyn\_score\_total：整体耗时
  - grasp\_matrix\_build
  - pinv (G+)
  - force\_generate
  - force\_check (friction cone + residual)
  - force\_eval (Emag/Edir/Evar + normalize + max)

> 说明：force\_generate / force\_check / pinv 是常见瓶颈，应强制可测。

### 5.3 统一事件模型（推荐）

每次测到一个区间，输出：

```json
{
  "ts": 1730000000.123,
  "run_id": "2026-01-06T16:00:00+09:00",
  "timestep": 12,
  "component": "cpp",
  "section": "dyn_score/force_generate",
  "duration_ms": 3.42,
  "meta": {"candidate_count": 1500, "force_samples": 1000}
}
```

- run\_id：一次运行唯一标识
- timestep：轨迹的 step index（无轨迹则为 -1）
- meta：记录规模（候选数、点数、采样数），用于做耗时与规模的相关性分析。

### 5.4 实现建议

#### 5.4.1 Python：Context Manager

- 以 `with timer("python/mesh_boolean", timestep=t):` 包裹。
- enable=false 时，timer 直接返回空上下文。

#### 5.4.2 C++：RAII ScopedTimer + section 开关

- `if (!Enable) return;`
- `ScopedTimer _(sink, "cpp/dyn_score/force_generate", meta);`

#### 5.4.3 配置驱动

- Python 解析 instrumentation.sections，并传入 C++（setConfig）。
- C++ 内部通过 `unordered_map<string,bool>` 或 enum bitmask 快速判断。

---

## 6. 开发注意事项（复现一致性与稳定性）

1. 坐标系一致性：open3d / trimesh / C++ 全部使用 world frame。
2. 数据生命周期：points\_low / normals\_low 必须在 C++ 对象存活期间保持有效（零拷贝）。
3. 数值稳定性：
   - 所有可能除零/反三角域外/伪逆病态都必须 `isfinite` 检查。
   - 将失败视为“合法淘汰”，统一映射为 \(-\infty\)。
4. 并行安全：OpenMP 下写入必须线程安全（每候选独立写入向量更容易）。
5. 复杂度提示：\(C_M^N\) 指数爆炸，N 建议 2–3。若要 N>=4，必须改变候选生成策略（例如采样/启发式）。

---

## 7. 验收标准（Definition of Done）

- 功能一致：
  - 能复现论文 Algorithm 1–4 的结果趋势（至少相同输入下输出稳定且可解释）。
- 性能可测：
  - instrumentation.enable\_timing=true 时，输出完整 JSONL，含每 step 的关键 section。
  - enable\_timing=false 时，性能回退不超过可接受阈值（建议 <1%）。
- 可调试：
  - 可选输出 ScoreDebugInfo 或关键中间量（仅 debug 模式）。
- 数值健壮：
  - 不因 NaN/Inf 崩溃；所有异常候选被合理淘汰。

---

> 注：本 Rev2 已将“算法不变式、失败模式、几何/物理解耦、归一化作用域、细粒度测时”纳入规格书主结构，便于后续直接生成代码骨架与实验对照。

