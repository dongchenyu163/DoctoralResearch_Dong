# WRONG_IMPL.md

记录与《Method Details》规范不符的实现问题，作为后续修复依据。

## 当前状态

截至 目前，`WRONG_IMPL` 中记录的问题已全部修复：

`dynamic score`的`Force Generation Method`计算有问题
原始的参考： @METHOD_DETAILS.md 中的 `#### C. Dynamic Score ($S_{force}$, Algorithm 4)`章节
- $f$是一个$3n$长度的列向量，存储着每根指头的力
- wrench就是论文中的$t_k$
1. **Force Generation Method (Proposed):**我给的不是很明确，这一大步需要执行很多次，用来生成很多组各个指头的力$f$
2. 这个执行（尝试生成）次数上限（不是最终生成的各个指头的力$f$的数量，只是尝试生成的次数）放到JSON中配置（默认到600）
3. `**Force Generation Method (Proposed):**`章节中的前三步是生成，最后一步要检查2个条件（同时满足）：
	- 条件1:生成的$f$是否可以平衡$t_k$：计算残差向量 G*f + wrench，衡量抓取平衡性（就是你原来的程序），残差向量的长度来判断是否平衡，具体的阈值通过JSON来配置
	- 条件2:是否在摩擦圆锥中（可以简化为一个通过JSON配置的圆锥角）
4. `检查是否在摩擦圆锥中` 如果失败那就继续生成，直到尝试次数用尽。
5. 尝试次数用尽，$f$的个数为0，则代表当前的握持候补$P$
6. 补充：这个函数对所有握持候补$P$来计算`dynamic score`，一共是2层嵌套循环：
	- 第一层是循环`握持候补$P$`（这个循环看上去没问题），
	- 第二层循环有好几个：
		- 第一个是`生成很多组各个指头的力$f$`的循环（为了执行`**Force Generation Method (Proposed):**`）
		- 如果第一个`生成很多组$f$`的循环生成个数为0（即所有尝试生成的$f$都不满足`3`中的两个检查条件），则结束这个第二层循环，并且把`握持候补$P$`的分数置为`非法`代表之后的所有计算都不参与了
		- 如果第一个`生成很多组$f$`的循环生成了很多组$f$，则循环生成的多组$f$，对每个$f$计算三项评分 `Magnitude` `Direction` `Variance`，也就是 `**Scoring (for a valid $f$):**`中的内容
		- 所有评分中得分最高的$f$的分数作为当前`握持候补$P$`的`dynamic score`输出出去。
7. 补充：为了调试的目的类似于【`filterByGeoScore`中把大部分计算的原始结果保存到成员变量`last_geo_order_`里面】，方便调试查看，所以
	- 为每个`握持候补$P$`保存所有`生成的多组$f$`（不论`**Force Generation Method (Proposed):**`的结果是否合法，也就是说数量应该是和`尝试生成的次数`相同）以及`生成的多组$f$`的三个评分（按照各自`$f$`评分总和进行排序，降序）
	- 要求是std::tuple把`$f$`和`$f$`的三个评分组合到一起，然后用std::vector来保存并按照评分排序。

### 你的**Force Generation Method (Proposed):**代码是
```C++
// 求解最小二乘问题 G*f = -wrench，得到每个接触点的力向量
Eigen::VectorXd f = G.completeOrthogonalDecomposition().solve(-wrench);

// 计算残差向量 G*f + wrench，衡量抓取平衡性
Eigen::VectorXd residual_vec = G * f + wrench;
double residual = residual_vec.norm();
```

### 参考
- **Force Generation Method (Proposed):**
    1.  Randomly generate $f_{init}$ inside friction cone.
    2.  Calculate residual needed: $t = (-t_k) - G f_{init}$.
    3.  Adjust: $f = f_{init} - G^+ t$ ($G^+$ is pseudo-inverse).
    4.  Check if $f$ is within friction cone.
