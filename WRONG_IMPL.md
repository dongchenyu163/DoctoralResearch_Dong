# WRONG_IMPL.md

记录与《Method Details》规范不符的实现问题，作为后续修复依据。

## 当前状态

截至 目前，`WRONG_IMPL` 中记录的问题已全部修复：

compute_wrench的计算有问题：

1. 断裂力和摩擦力的计算中的食材重心没有考虑：断裂力和摩擦力的计算都需要知道重心（@METHOD_DETAILS.md 文件中`#### B. Knife Wrench Calculation`章节的对应的计算公式用 $g$ 表示重心）；程序中使用高密度原始点云的中心（所有点平均位置）作为食材的重心。
2. 断裂力和摩擦力的方向没有考虑：应当和轨迹数据中的速度信息相关（@METHOD_DETAILS.md 文件中`#### B. Knife Wrench Calculation`章节的对应的计算公式中的 $\hat{v}$ 来表示速度的单位向量，也就是速度的方向）
3. 断裂力和摩擦力的扭矩的积分计算没有乘以$du$和$dS$：$du$是刀刃的step长度，$dS$是摩擦接触面的面片面积。
4. 摩擦力的方向：除了考虑速度方向，还要把速度方向投影到2个接触面上（因为原始的速度方向是在刀的中心面上的），具体的计算方法写在了下面的`### 4 的参考`


### 4 的参考
```latex
For the knife's velocity $\bm{v}$, let $\bm{v}_{\mathrm{c}j}$ be the velocity component parallel to the contact surface $\Omega_{\mathrm{c}j}$.
Given the normal vector $\hat{\bm{n}}_{\mathrm{c}j}$ of contact surface $\Omega_{\mathrm{c}j}$, this velocity component is calculated as $\bm{v}_{\mathrm{c}j} = \bm{v} - (\bm{v} \cdot \hat{\bm{n}}_{\mathrm{c}j})\,\hat{\bm{n}}_{\mathrm{c}j}$.
```

