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
        surface = ContactSurfaceResult(faces=faces, metadata={"total_faces": 2})
        wrench = compute_wrench(surface, config)
        self.assertEqual(wrench.shape, (6,))
        self.assertAlmostEqual(wrench[2], 0.0)
        self.assertAlmostEqual(wrench[3], 0.0)
        self.assertAlmostEqual(wrench[4], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
