# WRONG_IMPL.md

记录与《Method Details》规范不符的实现问题，作为后续修复依据。

## 当前状态

截至 目前，`WRONG_IMPL` 中记录的问题已全部修复：

1. **接触面 / Knife Wrench 每步更新**  
   - `extract_contact_surface` 现已接收 `knife_pose` 并在 `for step` 循环内调用；`compute_wrench` 同步运行，`pipeline_stub` 使用最新 wrench 参与 Algorithm 4 计算。

2. **Ωg 动态依赖刀具位姿**  
   - `compute_valid_indices` 现在接受 `knife_point`+`knife_normal` 并新增平面过滤；轨迹循环每个 step 都重新计算 Ωg。

如发现新的偏差，请在此文件继续登记。
