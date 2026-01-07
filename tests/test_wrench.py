"""Tests for the wrench computation helpers."""

from __future__ import annotations

import unittest

import numpy as np

from python.pipeline.contact_surface import ContactSurfaceResult
from python.pipeline.wrench import compute_wrench
from tests.helpers import make_config


class WrenchTests(unittest.TestCase):
    def test_compute_wrench_planar_constraint(self) -> None:
        config = make_config({"physics": {"planar_constraint": True}})
        faces = [
            np.array(
                [
                    [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
                ]
            )
        ]
        surface = ContactSurfaceResult(faces=faces, metadata={"total_faces": 2}, mesh=None)
        wrench = compute_wrench(surface, config, velocity=np.array([0.0, 0.0, 1.0]))
        self.assertEqual(wrench.shape, (6,))
        self.assertAlmostEqual(wrench[2], 0.0)
        self.assertAlmostEqual(wrench[3], 0.0)
        self.assertAlmostEqual(wrench[4], 0.0)

    def test_friction_contribution_changes_force(self) -> None:
        base_config = make_config({"physics": {"planar_constraint": False}})
        faces = [
            np.array(
                [
                    [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
                ]
            )
        ]
        surface = ContactSurfaceResult(faces=faces, metadata={}, mesh=None)
        config_zero = make_config({"physics": {"friction_coef": 0.0, "planar_constraint": False}})
        config_high = make_config({"physics": {"friction_coef": 1.0, "planar_constraint": False}})
        velocity = np.array([1.0, 0.0, 0.0])
        wrench_zero = compute_wrench(surface, config_zero, velocity=velocity)
        wrench_high = compute_wrench(surface, config_high, velocity=velocity)
        self.assertTrue(np.linalg.norm(wrench_high[:3]) > np.linalg.norm(wrench_zero[:3]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
