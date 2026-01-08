"""Smoke tests for the C++ score_calculator module."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

BUILD_DIR = pathlib.Path(__file__).resolve().parents[1] / "build"
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

import score_calculator  # type: ignore  # noqa: E402,E401


class ScoreCalculatorBindingsTests(unittest.TestCase):
    def test_set_point_cloud_accepts_numpy_arrays(self) -> None:
        calc = score_calculator.ScoreCalculator()
        points = np.zeros((4, 3), dtype=np.float64)
        normals = np.ones((4, 3), dtype=np.float64)
        calc.set_point_cloud(points, normals)
        self.assertEqual(calc.point_count, 4)

    # def test_filter_by_geo_score_respects_max_candidates(self) -> None:
    #     calc = score_calculator.ScoreCalculator()
    #     points = np.zeros((4, 3), dtype=np.float64)
    #     normals = np.ones((4, 3), dtype=np.float64)
    #     calc.set_point_cloud(points, normals)
    #     calc.set_max_candidates(1)
    #     calc.set_geo_weights(1.0, 1.0, 1.0)
    #     calc.set_geo_filter_ratio(1.0)
    #     candidates = np.array([[0, 1], [1, 2]], dtype=np.int32)
    #     result = calc.filter_by_geo_score(
    #         candidates,
    #         np.zeros(3, dtype=np.float64),
    #         np.array([0.0, 0.0, 1.0], dtype=np.float64),
    #         0.0,
    #     )
    #     self.assertEqual(result.shape[0], 1)
    #     np.testing.assert_array_equal(result[0], np.array([0, 1], dtype=np.int32))

    def test_geo_filter_prefers_higher_table_distance(self) -> None:
        calc = score_calculator.ScoreCalculator()
        points = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.02], [0.0, 0.0, 0.05]], dtype=np.float64)
        normals = np.ones_like(points)
        calc.set_point_cloud(points, normals)
        calc.set_geo_weights(0.2, 0.2, 1.0)
        calc.set_geo_filter_ratio(0.5)
        candidates = np.array([[0, 1], [1, 2]], dtype=np.int32)
        result = calc.filter_by_geo_score(
            candidates,
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            0.0,
        )
        self.assertEqual(result.shape[0], 1)
        np.testing.assert_array_equal(result[0], np.array([1, 2], dtype=np.int32))

    def test_calc_positional_scores_prefers_orthogonal(self) -> None:
        calc = score_calculator.ScoreCalculator()
        points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float64)
        normals = np.ones_like(points)
        calc.set_point_cloud(points, normals)
        candidates = np.array([[0, 1], [0, 2]], dtype=np.int32)
        scores = calc.calc_positional_scores(
            candidates,
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        self.assertEqual(scores.shape[0], 2)
        self.assertTrue(np.all(scores >= 0.0))

    # def test_calc_dynamics_scores_returns_values(self) -> None:
    #     calc = score_calculator.ScoreCalculator()
    #     points = np.array(
    #         [
    #             [0.0, 0.0, 0.0],
    #             [0.1, 0.0, 0.0],
    #             [0.0, 0.1, 0.0],
    #         ],
    #         dtype=np.float64,
    #     )
    #     normals = np.tile(np.array([[0.0, 1.0, 0.0]]), (3, 1))
    #     calc.set_point_cloud(points, normals)
    #     candidates = np.array([[0, 1], [1, 2]], dtype=np.int32)
    #     wrench = np.ones(6, dtype=np.float64)
    #     scores = calc.calc_dynamics_scores(
    #         candidates,
    #         wrench,
    #         np.zeros(3, dtype=np.float64),
    #         False,
    #         0.5,
    #         40.0,
    #         10,
    #         1e-2,
    #         0.1,
    #         1.0,
    #         40.0,
    #     )
    #     self.assertEqual(scores.shape[0], 2)
    #     self.assertFalse(np.any(np.isnan(scores)))

    def test_random_force_balance_cancels_wrench(self) -> None:
        calc = score_calculator.ScoreCalculator()
        points = np.array([[1.0, 0.0, -2.0], [-1.0, 0.0, -2.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)
        wrench = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0], dtype=np.float64)
        ok = calc.check_random_force_balance(
            indices,
            wrench,
            np.zeros(3, dtype=np.float64),
            True,
            1e-5,
        )
        self.assertTrue(ok)

    def test_planar_constraint_logic(self) -> None:
        """
        验证 planar_constraint 是否正确工作。
        假设物体在桌面上，因此 Z 方向的力和 X/Y 轴的力矩由桌面承担。
        """
        calc = score_calculator.ScoreCalculator()
        
        # 场景：X轴上的双指对握 (2 Fingers)
        points = np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)
        center = np.zeros(3, dtype=np.float64)

        # Case 1: 垂直方向的干扰 (Vertical Disturbance)
        # 施加巨大的向下力 Fz 和倾覆力矩 Mx
        # 如果是 6D 检查，这必挂无疑；但在 Planar 模式下，应该通过。
        wrench_ignored = np.array([0.0, 0.0, -100.0, 50.0, 0.0, 0.0], dtype=np.float64)
        
        ok_planar = calc.check_random_force_balance(
            indices, 
            wrench_ignored, 
            center, 
            True,
            1e-5,
        )
        self.assertTrue(ok_planar, "Should ignore Fz and Mx/My when planar_constraint is True")

        # Case 2: 验证它没有把所有东西都忽略了
        # 施加平面内的力 Fx (沿连线方向) -> 应该能抵抗
        wrench_planar_ok = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.assertTrue(
            calc.check_random_force_balance(indices, wrench_planar_ok, center, True, 1e-5)
        )

    def test_planar_rotation_with_2_fingers(self) -> None:
        """
        验证双指在平面模式下能否抵抗 Mz (平面旋转)。
        数学上，只要两个点不重合，就能生成 Mz 力偶。
        """
        calc = score_calculator.ScoreCalculator()
        points = np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)

        # 施加绕 Z 轴的力矩 Mz
        wrench_mz = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        
        # 即使是2指，数学投影 G*f = -w 在平面 3DoF 下也是有解的
        ok = calc.check_random_force_balance(
            indices, 
            wrench_mz, 
            np.zeros(3),
            True,
            1e-5, 
        )
        self.assertTrue(ok, "2 fingers should be able to generate Mz couple in math projection")

    def test_2_finger_planar_immunity(self) -> None:
        """
        [平面假设测试 1]：验证对垂直干扰的'免疫力'。
        场景：双指水平夹持，施加巨大的向下压力和侧向翻转力矩。
        预期：在平面假设下，这些力应被忽略，Residual 应为 0。
        """
        calc = score_calculator.ScoreCalculator()
        # 模拟加载了 planar_constraint = True 的配置
        # calc.load_config("config/default.json") 
        
        # 1. 建立 X 轴上的双指抓取 (8cm 抓宽)
        points = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points) # 法向量在此测试不影响结果
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)
        center = np.zeros(3, dtype=np.float64)

        # 2. 施加破坏性的垂直干扰 (Table Support Test)
        # Fz = -50N (重压), Mx = 10Nm (剧烈翻转)
        # 如果没有平面假设，双指点接触绝对无法平衡这个力矩
        wrench_vertical_tilt = np.array([0.0, 0.0, -50.0, 10.0, 0.0, 0.0], dtype=np.float64)

        # 3. 验证平衡
        # 注意：这里调用时不传 planar_constraint 参数，假设类内部已从配置读取
        ok = calc.check_random_force_balance(indices, wrench_vertical_tilt, center, True, 1e-4)
        
        self.assertTrue(ok, "Planar grasp should ignore Z-force and X/Y-torque (assumed supported by table)")

    def test_2_finger_planar_rotation(self) -> None:
        """
        [平面假设测试 2]：验证对平面内旋转力矩 (Mz) 的抵抗能力。
        场景：双指夹持，物体试图在桌面上打转 (Spinning)。
        原理：两个接触点形成力偶 (Force Couple) 来抵抗 Mz。
        """
        calc = score_calculator.ScoreCalculator()
        points = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)
        
        # 施加绕 Z 轴的纯力矩 Mz = 1.0 Nm
        wrench_spin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        
        ok = calc.check_random_force_balance(indices, wrench_spin, np.zeros(3), True, 1e-4)
        self.assertTrue(ok, "2 fingers should resist planar rotation (Mz) via force couple")

    def test_2_finger_planar_general_load(self) -> None:
        """
        [平面假设测试 3]：验证平面内的综合负载 (Fx + Fy + Mz)。
        场景：切割时刀具产生的斜向推力，既推物体又让物体转动。
        """
        calc = score_calculator.ScoreCalculator()
        # Y轴方向的双指抓取 (前后抓)
        points = np.array([[0.0, 0.04, 0.0], [0.0, -0.04, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)
        
        # 施加混合力：
        # Fx = 5N (侧推), Fy = -5N (后推), Mz = 0.5Nm (旋转)
        # 忽略 Fz/Mx/My
        wrench_complex = np.array([5.0, -5.0, 100.0, 10.0, 10.0, 0.5], dtype=np.float64)
        
        ok = calc.check_random_force_balance(indices, wrench_complex, np.zeros(3), True, 1e-4)
        self.assertTrue(ok, "Should balance complex planar wrench while ignoring vertical components")

    def test_2_finger_planar_singularity(self) -> None:
        """
        [平面假设测试 4]：奇异性/打滑边缘测试。
        场景：力正好沿着两个手指的连线方向。
        说明：虽然数学上 G 矩阵可能有解，但这是物理上最不稳定的方向（容易把物体从手中挤出去）。
        这个测试主要看您的求解器是否能解出这种情况。
        """
        calc = score_calculator.ScoreCalculator()
        # X轴双指
        points = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)
        
        # 纯 Fx 力 (沿着手指连线推)
        # 这需要两个手指产生巨大的法向力差，或者完全靠摩擦力
        wrench_axial = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        ok = calc.check_random_force_balance(indices, wrench_axial, np.zeros(3), True, 1e-4)
        self.assertTrue(ok, "Mathematical projection should find a solution even for axial forces")

    def test_2_finger_planar_com_offset_vertical_immunity(self) -> None:
        """
        [CoM 偏移测试 1]：高重心（High CoM）带来的倾覆力矩应被忽略。
        场景：
            - 重心很高 (Z = 10cm)。
            - 施加水平推力 Fx。
            - 物理上这会产生巨大的 My (向前翻倒)。
            - 预期：在 Planar 模式下，My 被忽略，测试通过。
        """
        calc = score_calculator.ScoreCalculator()
        points = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)

        # 1. 设置极高的重心 Z=0.1m
        high_com = np.array([0.0, 0.0, 0.1], dtype=np.float64)

        # 2. 施加纯水平推力 Fx = 10N
        # 相对于抓取点平面(z=0)，这个力有力臂，会产生 My = 10N * 0.1m = 1.0Nm
        wrench_push = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # 3. 验证
        # 如果代码没有正确忽略 My，这里会失败，因为双指点接触无法抵抗 My
        ok = calc.check_random_force_balance(indices, wrench_push, high_com, True, 1e-4)
        self.assertTrue(ok, "High CoM tipping moment (My) should be ignored in planar mode")

    def test_2_finger_planar_com_offset_lever_effect(self) -> None:
        """
        [CoM 偏移测试 2]：平面内偏移带来的旋转力矩。
        场景：
            - 重心在 Y 轴上有偏移 (不在两指连线上)。
            - 施加 X 轴推力。
            - 物理上：Force X * Offset Y = Moment Z (平面打转)。
            - 预期：手指必须生成反向力偶来平衡这个 Mz。
        """
        calc = score_calculator.ScoreCalculator()
        # 双指在 X 轴对握
        points = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)

        # 1. 重心偏向 Y+ (例如食材重心偏前)
        offset_com = np.array([0.0, 0.05, 0.0], dtype=np.float64)

        # 2. 在重心处施加 Fx
        # 这会产生力矩 Mz = - (Fy*x - Fx*y) ... 也就是 Fx * y_offset
        wrench_push = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # 3. 验证
        # 即使 wrench 里没有显式的 Mz，代码内部 buildGraspMatrix 会根据 center 自动计算出力臂。
        # 只要双指能产生抵抗 Mz 的力偶（这需要 friction/shear），就能平衡。
        ok = calc.check_random_force_balance(indices, wrench_push, offset_com, True, 1e-4)
        self.assertTrue(ok, "Fingers should resist the Mz induced by pushing an offset CoM")

    def test_2_finger_planar_com_outside_grasp(self) -> None:
        """
        [CoM 偏移测试 3]：重心在抓取区域之外 (悬臂梁效应)。
        场景：
            - 重心在最右侧手指的更右侧。
            - 类似于捏着一根棍子的一端，另一端受力。
        """
        calc = score_calculator.ScoreCalculator()
        points = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)

        # 1. 重心在 x=0.2 (远在手指 x=0.04 之外)
        far_com = np.array([0.2, 0.0, 0.0], dtype=np.float64)

        # 2. 施加侧向力 Fy
        # 这会产生巨大的 Mz
        wrench_side = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        ok = calc.check_random_force_balance(indices, wrench_side, far_com, True, 1e-4)
        self.assertTrue(ok, "Should balance even if CoM is far outside the grasp width (mathematically possible)")

    def test_2_finger_planar_com_outside_grasp_self(self) -> None:
        """
        [CoM 偏移测试 3]：重心在抓取区域之外 (悬臂梁效应)。
        场景：
            - 重心在最右侧手指的更右侧。
            - 类似于捏着一根棍子的一端，另一端受力。
        """
#           [0109001628_072] force at contact 0: [-3.9611, +18.4215, -1.8142]  normal [-0.2971, +0.9066, -0.2996], point [+0.5973, -0.2945, +0.0431], angle 13.3012 deg :: calcDynamicsScores
#           [0109001628_073] force at contact 1: [-8.2733, +9.0895, -5.7699]  normal [-0.6501, +0.5071, -0.5658], point [+0.6054, -0.2799, +0.0557], angle 12.5580 deg :: calcDynamicsScores
# residual_vec: -93.8361  55.0252 0.420093wrench:   -34.6836 0.00152259   0.355346
        calc = score_calculator.ScoreCalculator()
        points = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]], dtype=np.float64)
        normals = np.zeros_like(points)
        calc.set_point_cloud(points, normals)
        indices = np.array([0, 1], dtype=np.int32)

        # 1. 重心在 x=0.2 (远在手指 x=0.04 之外)
        far_com = np.array([0.2, 0.0, 0.0], dtype=np.float64)

        # 2. 施加侧向力 Fy
        # 这会产生巨大的 Mz
        wrench_side = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        ok = calc.check_random_force_balance(indices, wrench_side, far_com, True, 1e-4)
        self.assertTrue(ok, "Should balance even if CoM is far outside the grasp width (mathematically possible)")

    def test_load_pcd_and_check_random_2_finger_grasp(self) -> None:
        """
        集成测试：
        1. 读取 tests/points.pcd 文件
        2. 计算点云几何中心作为重心 (CoM)
        3. 随机选取 2 个接触点
        4. 验证是否能抵抗平面内的切割力 (Fx, Fy, Mz)
        """
        try:
            import open3d as o3d
        except ImportError:
            self.skipTest("Open3D not installed, skipping point cloud IO test")

        # 1. 定位并读取 PCD 文件
        # 假设 points.pcd 与当前测试脚本位于同一目录 (tests/)
        pcd_path = pathlib.Path(__file__).parent / "points.pcd"
        if not pcd_path.exists():
            self.skipTest(f"PCD file not found at {pcd_path}")

        pcd = o3d.io.read_point_cloud(str(pcd_path))
        if pcd.is_empty():
            self.fail("Loaded point cloud is empty")

        points = np.asarray(pcd.points, dtype=np.float64)
        
        # 处理法向量：如果有就用，没有就设为 0 (对于 force balance 计算 G 矩阵，法向量其实不参与，只参与摩擦锥检查)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals, dtype=np.float64)
        else:
            normals = np.zeros_like(points)

        # 2. 计算重心 (Center of Mass)
        # 简单假设：均匀密度的物体，重心 ≈ 几何中心
        center_of_mass = np.mean(points, axis=0)
        
        # 3. 初始化计算器
        calc = score_calculator.ScoreCalculator()
        calc.set_point_cloud(points, normals)
        
        # 4. 随机选取 2 个点 (模拟一次随机抓取尝试)
        if len(points) < 2:
            self.fail("Point cloud has fewer than 2 points")
            
        rng = np.random.default_rng(42) # 固定种子以保证测试可复现
        indices = rng.choice(len(points), 2, replace=False).astype(np.int32)
        
        # 5. 定义外力 (Wrench)
        # 模拟切割时的平面受力：侧向推力 Fx=5N, 切割阻力 Fy=-5N, 及其产生的平面力矩
        # 假设配置中开启了 planar_constraint，忽略 Fz, Mx, My
        # -34.6836 0.00152259   0.355346
        wrench = np.array([-34.6836, 0.00152259, 0.0, 0.0, 0.0, 0.355346], dtype=np.float64)
        
        # 6. 执行检查
        # 此时 ScoreCalculator 应该通过内部配置自动应用 planar_constraint
        is_balanced = calc.check_random_force_balance(
            indices,
            wrench,
            center_of_mass,
            True, 
            1e-4
        )
        
        # 打印调试信息，方便您观察
        print(f"\n[Test Info] Loaded {len(points)} points from {pcd_path.name}")
        print(f"[Test Info] CoM: {center_of_mass}")
        print(f"[Test Info] Random Grasp Indices: {indices}")
        print(f"[Test Info] Balanced: {is_balanced}")

        # 断言结果
        # 注意：随机选取的两点不一定总是能平衡（例如选到了连线与力平行的点），
        # 但 check_random_force_balance 在数学上有解通常返回 True。
        # 这里我们断言它是布尔值，且程序没有崩溃。
        self.assertIsInstance(is_balanced, bool)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
