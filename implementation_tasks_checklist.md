# 实装任务清单（GitHub Issues 风格）

> 目的：把《实装步骤与计划文档》进一步拆解为可直接创建 Issues 的任务列表。
>
> 结构：Phase → Issue（子任务）→ Checklist（可勾选）→ 验收标准（DoD）→ 埋点位置（instrumentation sections）
>
> 备注：你已提供可参考的 CMake 配置（用于本地已安装库的 find/link + FetchContent + pybind11 + OpenMP 等），C++ 构建相关的 Issue 直接以该文件为模板拷贝改名即可。&#x20;

---

## 运行环境前置要求（全局适用）

> **重要约定**：所有 Python 可执行入口（`python/main.py`、`bench/bench_run.py`、测试脚本等）在执行前，**必须先切换到名为 ****************************************************************************************************************`new_env_testing`**************************************************************************************************************** 的 Python 环境**。
>
> 该环境用于保证依赖版本、pybind11 扩展、OpenMP 运行时的一致性，避免污染系统或其他研究环境。

### 统一执行规范（示例） &#x20;

所有 Python 入口均 \*\*必须\*\* 通过 \`pyenv activate\` 切换环境。

&#x20;&#x20;

- pyenv（Win / Linux / macOS）：
  ```bash
  pyenv activate new_env_testing
  python -m python.main --config configs/default.json
  ```

---

# **实装任务清单 (Implementation Roadmap v2)**

**版本说明**：基于 V1 版本修正，整合了最新的 C++ 数据结构优化 (Eigen::Matrix) 和接触面法向过滤算法，并新增了 Master TodoList。

## **📅 Master TodoList (总览看板)**

将此列表复制到项目的 README.md 或 GitHub Issue \#1 中以追踪进度。

* \[ \] **Phase 0: 工程骨架 (Skeleton)**  
  * \[ \] P0-1: 目录结构与最小入口 (python/main.py)  
  * \[ \] P0-2: 埋点日志系统 (Instrumentation)  
  * \[ \] P0-3: 随机种子复现性  
* \[ \] **Phase 1: 数据预处理 (Preprocessing)**  
  * \[ \] P1-1: 点云降采样 & 法向估计  
  * \[ \] P1-2: 组合索引全集生成 ($C\_M^N$)  
  * \[ \] P1-3: 有效区域 ($\\Omega\_g$) 索引掩码计算  
* \[ \] **Phase 2: 几何筛选 (C++ GeoFilter)**  
  * \[ \] **C-1**: CMake 构建系统搭建 (Eigen3 \+ pybind11 \+ OpenMP)  
  * \[ \] P2-1: C++ 核心类骨架 & 数据绑定 (Eigen Ref)  
  * \[ \] P2-2: 几何评分实现 ($E\_{fin}, E\_{knf}, E\_{tbl}$) & Top-K 筛选  
* \[ \] **Phase 3: 位置评分 (C++ PosScore)**  
  * \[ \] P3-1: 方向评分 ($E\_{pdir}$)  
  * \[ \] P3-2: 力臂评分 ($E\_{pdis}$) & 归一化  
* \[ \] **Phase 4: 包丁力计算 (Python Physics)**  
  * \[ \] P4-1: 接触面获取 (Trimesh Boolean \+ 法向过滤优化)  
  * \[ \] P4-2: 简化版 Wrench 计算  
  * \[ \] P4-3: 完整版 Wrench (Fracture \+ Friction)  
* \[ \] **Phase 5: 动力学评分 (C++ DynScore)**  
  * \[ \] P5-1: Grasp Matrix 构造  
  * \[ \] P5-2: 伪逆求解与力生成 (Sampling)  
  * \[ \] P5-3: 力修正与平衡校验 (Correction & Residual Check)  
  * \[ \] P5-4: 动力学评分 ($E\_{mag}, E\_{dir}, E\_{var}$) & 归一化  
* \[ \] **Phase 6: 流程整合 (Integration)**  
  * \[ \] P6-1: 完整 Algorithm 1 循环对齐  
  * \[ \] P6-2: 结果输出与调试 Dump  
* \[ \] **Phase 7: 验证与文档 (Validation)**  
  * \[ \] P7-1: 性能基准测试 (Benchmark)  
  * \[ \] P7-2: 文档与使用指南

## **详细任务列表 (Detailed Issues)**

### **Phase 0 — 工程骨架 \+ 可运行空管线（M0）**

#### **Issue P0-1：建立 repo 目录结构与最小可运行入口**

**验收标准 (DoD)**

* python \-m python.main \--config configs/default.json 能跑通。  
* 生成空的 output/result.json 和 logs/timing.jsonl。  
* 确保 config.json 包含所有 Section (weights, physics, etc.) 的默认值。

#### **Issue P0-2：实现 instrumentation 框架**

**验收标准 (DoD)**

* 实现基于 Context Manager 的计时器，支持 JSONL 流式写入。  
* 能够统计 Python 端和 C++ 端（通过累计耗时）的性能数据。

#### **Issue P0-3：随机种子与可复现性**

**验收标准 (DoD)**

* 固定 Seed 后，Python 的 random、numpy.random 以及 C++ 侧（若有随机数生成）的行为一致。

### **Phase 1 — Python 侧预处理（M1）**

#### **Issue P1-1：点云降采样与法向估计**

**说明**

* $\\Omega\_{low}$ 生成后在整个生命周期不变。  
  验收标准 (DoD)  
* 输入点云后，能正确生成 points\_low (Nx3) 和 normals\_low (Nx3) 的 numpy 数组。

#### **Issue P1-2：组合全集索引生成**

**说明**

* 生成 $C\_M^N$ 的索引列表。  
  验收标准 (DoD)  
* 生成 itertools.combinations 结果或等价的 numpy 索引数组。  
* **注意**：此处仅在 Python 端生成逻辑概念，实际传给 C++ 时需考虑内存布局（见 P2-1）。

#### **Issue P1-3：$\\Omega\_g$ (Valid Indices) 计算**

**验收标准 (DoD)**

* 根据刀具平面方程，正确判断点在平面哪一侧。  
* 返回一个 valid\_indices 列表或布尔掩码。

### **Phase 2 — GeoFilter（Algorithm 2）C++ 并行实现（M2）**

#### **Issue C-1：CMake 构建环境搭建**

**说明**

* 配置 CMakeLists.txt，引入 Eigen3, OpenMP, pybind11。  
* 定义编译期常量 constexpr int FINGER\_COUNT \= 2;。  
  验收标准 (DoD)  
* pip install . 或 python setup.py build\_ext \--inplace 成功编译扩展。

#### **Issue P2-1：C++ ScoreCalculator 骨架与数据绑定**

**说明**

* 实现 setPointCloud，使用 Eigen::Ref 实现零拷贝接收 Numpy 数组。  
* 定义内部数据结构 CandidateMatrix (Eigen动态行，固定列)。  
  验收标准 (DoD)  
* Python 能调用 setPointCloud 且数据在 C++ 端正确读取（打印前几个点验证）。

#### **Issue P2-2：GeoFilter 实现 ($E\_{fin}, E\_{knf}, E\_{tbl}$)**

**说明**

* **修正**：输入改为 valid\_point\_indices，C++ 内部根据此列表生成/过滤候选组合。  
* 使用 Eigen::Matrix 进行列向量化计算归一化。  
  验收标准 (DoD)  
* 并行计算所有组合的几何分。  
* 输出 Top-K 个候选组合的索引矩阵。  
* **鲁棒性**：处理 NaN/Inf 情况。

### **Phase 3 — PosScore（Algorithm 3）C++ 实现（M3）**

#### **Issue P3-1：PosScore 实现 ($E\_{pdir}, E\_{pdis}$)**

**验收标准 (DoD)**

* 接收 P2 输出的候选索引矩阵。  
* $E\_{pdir}$: 实现 PCA 或向量夹角计算。  
* $E\_{pdis}$: 计算点到刀平面的距离。  
* 返回分数向量 Eigen::VectorXd 对应的 Numpy 数组。

### **Phase 4 — Knife Wrench 接口打通（M4）**

#### **Issue P4-1：接触面获取 (Trimesh \+ 法向过滤)**

**说明**

* **修正**：不再使用投影填充。  
* 步骤：  
  1. trimesh.boolean.intersection 获取交集网格。  
  2. 计算每个面的法向，与刀具侧面法向做点积。  
  3. 剔除点积绝对值不接近 1 的面（即剔除食品外表面）。  
  4. 连通域分离 $\\Omega\_{c1}, \\Omega\_{c2}$。  
     验收标准 (DoD)  
* 能够正确提取出纯净的刀具接触面切片。

#### **Issue P4-2 & P4-3：Wrench 计算**

**验收标准 (DoD)**

* 实现积分计算（Fracture \+ Friction）。  
* 实现 planar\_constraint：强制置零非平面分量。

### **Phase 5 — DynScore（Algorithm 4）C++ 实现（M5）**

#### **Issue P5-1：Grasp Matrix & 力生成骨架**

**验收标准 (DoD)**

* 实现 Grasp Matrix $G$ 的构造。  
* 能够从 config 读取 friction\_cone 参数和 force\_sample\_count。

#### **Issue P5-2 & P5-3：采样、修正与校验**

**说明**

* **核心逻辑**：  
  1. 随机采样 $f\_{init}$。  
  2. 计算 $\\Delta t \= \-t\_k \- G f\_{init}$。  
  3. 修正 $f \= f\_{init} \+ G^+ \\Delta t$。  
  4. 校验：$f$ 是否在摩擦锥内？$\\|Gf \+ t\_k\\|$ 是否小于阈值？  
     验收标准 (DoD)  
* 只有通过校验的力才参与评分。

#### **Issue P5-4：评分与归一化**

**验收标准 (DoD)**

* 计算 $E\_{mag}, E\_{dir}, E\_{var}$。  
* 返回最终的 S\_dyn 向量。

### **Phase 6 & 7 — 整合与收尾**

#### **Issue P6-1：主循环整合**

**验收标准 (DoD)**

* 串联 Python 预处理 \-\> C++ GeoFilter \-\> C++ PosScore \-\> Python Wrench \-\> C++ DynScore。  
* 确保索引对齐无误。

#### **Issue P7-1：Benchmark**

**验收标准 (DoD)**

* 运行完整流程，记录不同 downsample\_num 和 force\_sample\_count 下的耗时。

