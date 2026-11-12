#!/usr/bin/env python3
"""Unit tests for automated aerodynamic polar tools: run_polar.py and generate_naca0012.py."""

import math
import tempfile
import unittest
from pathlib import Path
import sys

# Ensure tools directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from run_polar import (
    transform_forces_to_wind,
    parse_monitors_file,
    update_control_file,
    update_physical_file,
    prepare_warm_start_var,
)
from generate_naca0012 import naca0012_yt, generate_naca0012_su2


class TestForceTransformation(unittest.TestCase):
    def test_alpha_zero(self):
        # At alpha = 0: CL = CFy, CD = CFx
        res = transform_forces_to_wind(alpha_deg=0.0, fx=0.05, fy=0.50, mz=0.01)
        self.assertAlmostEqual(res["alpha_deg"], 0.0)
        self.assertAlmostEqual(res["alpha_rad"], 0.0)
        self.assertAlmostEqual(res["CL"], 0.50)
        self.assertAlmostEqual(res["CD"], 0.05)
        self.assertAlmostEqual(res["CM"], 0.01)
        self.assertAlmostEqual(res["L_over_D"], 10.0)

    def test_alpha_90(self):
        # At alpha = 90 deg:
        # CL = -fx * sin(90) + fy * cos(90) = -fx
        # CD =  fx * cos(90) + fy * sin(90) =  fy
        res = transform_forces_to_wind(alpha_deg=90.0, fx=0.20, fy=0.80)
        self.assertAlmostEqual(res["CL"], -0.20)
        self.assertAlmostEqual(res["CD"], 0.80)
        self.assertAlmostEqual(res["L_over_D"], -0.25)

    def test_alpha_45_symmetry(self):
        # At alpha = 45 deg: cos = sin = 1/sqrt(2)
        # If fx = 1, fy = 1: CL = (-1 + 1)/sqrt(2) = 0, CD = (1 + 1)/sqrt(2) = sqrt(2)
        res = transform_forces_to_wind(alpha_deg=45.0, fx=1.0, fy=1.0)
        self.assertAlmostEqual(res["CL"], 0.0)
        self.assertAlmostEqual(res["CD"], math.sqrt(2.0))

    def test_pressure_friction_decomposition(self):
        fx_p, fy_p = 0.03, 0.40
        fx, fy = 0.05, 0.40
        res = transform_forces_to_wind(
            alpha_deg=5.0,
            fx=fx,
            fy=fy,
            fx_press=fx_p,
            fy_press=fy_p,
        )
        # Total CD must equal CD_pressure + CD_friction
        self.assertAlmostEqual(res["CD"], res["CD_pressure"] + res["CD_friction"], places=10)
        # Total CL must equal CL_pressure + CL_friction
        self.assertAlmostEqual(res["CL"], res["CL_pressure"], places=1)


class TestMonitorParsing(unittest.TestCase):
    def test_parse_monitors_with_cm(self):
        content = """# Comment line 1
# Comment line 2
step time cd cl cm cd_surface cl_surface cm_surface cd_pressure cl_pressure cm_pressure
1 0.01 0.05 0.50 0.012 0.04 0.48 0.010 0.03 0.45 0.010
2 0.02 0.04 0.52 0.011 0.035 0.50 0.009 0.025 0.47 0.009
"""
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(content)
            f_path = Path(f.name)

        try:
            col_data, last_row = parse_monitors_file(f_path)
            self.assertIsNotNone(col_data)
            self.assertEqual(len(col_data["step"]), 2)
            self.assertAlmostEqual(last_row["step"], 2)
            self.assertAlmostEqual(last_row["cd"], 0.04)
            self.assertAlmostEqual(last_row["cl"], 0.52)
            self.assertAlmostEqual(last_row["cm"], 0.011)
            self.assertAlmostEqual(last_row["cm_pressure"], 0.009)
        finally:
            f_path.unlink()

    def test_parse_monitors_legacy_format(self):
        content = """# HydraMa2D monitors
step time cd cl cd_surface cl_surface cd_pressure cl_pressure
1 0.01 0.05 0.50 0.04 0.45 0.03 0.40
"""
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(content)
            f_path = Path(f.name)

        try:
            col_data, last_row = parse_monitors_file(f_path)
            self.assertIsNotNone(last_row)
            self.assertAlmostEqual(last_row["cd"], 0.05)
            self.assertAlmostEqual(last_row["cl"], 0.50)
            self.assertNotIn("cm", last_row)
        finally:
            f_path.unlink()


class TestControlAndPhysicalFileUpdates(unittest.TestCase):
    def test_update_control_file(self):
        # Simulate 15 blocks of 3 lines
        lines = []
        for i in range(15):
            lines.extend(["---\n", f"Header {i}\n", "0\n"])
        # Block 1 (ntime)
        lines[5] = "500\n"
        # Block 13 (tol)
        lines[41] = "1.000e-05\n"

        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.writelines(lines)
            f_path = Path(f.name)

        try:
            update_control_file(f_path, restart_opt=1, ntime=250, tol=1e-8)
            with open(f_path, "r") as f:
                res_lines = f.readlines()
            # Non-empty line 2: restart_opt
            self.assertEqual(res_lines[2].strip(), "1")
            # Non-empty line 5: ntime
            self.assertEqual(res_lines[5].strip(), "250")
            # Non-empty line 41: tol
            self.assertEqual(res_lines[41].strip(), "1.000e-08")
        finally:
            f_path.unlink()

    def test_update_physical_file(self):
        lines = [
            "---\n", "Reynolds Number\n", "100.0\n",
            "---\n", "pnode\n", "1\n",
        ]
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.writelines(lines)
            f_path = Path(f.name)

        try:
            update_physical_file(f_path, re_override=5000.0)
            with open(f_path, "r") as f:
                res_lines = f.readlines()
            self.assertEqual(res_lines[2].strip(), "5000.0000")
        finally:
            f_path.unlink()

    def test_prepare_warm_start_var(self):
        source_content = "10.5 500\n1.0 0.0 0.1\n0.9 0.1 0.1\n"
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write(source_content)
            src_path = Path(f.name)

        target_path = src_path.parent / (src_path.name + "_target")
        try:
            prepare_warm_start_var(src_path, target_path)
            with open(target_path, "r") as f:
                lines = f.readlines()
            self.assertEqual(lines[0].strip(), "0.0 0")
            self.assertEqual(lines[1].strip(), "1.0 0.0 0.1")
            self.assertEqual(lines[2].strip(), "0.9 0.1 0.1")
        finally:
            src_path.unlink()
            if target_path.exists():
                target_path.unlink()


class TestNaca0012MeshGenerator(unittest.TestCase):
    def test_thickness_profile(self):
        # Leading edge thickness is 0
        self.assertAlmostEqual(naca0012_yt(0.0), 0.0, places=6)
        # Max thickness near x = 0.3 should be ~0.06 (half thickness t = 0.12)
        y_max = naca0012_yt(0.3)
        self.assertGreater(y_max, 0.059)
        self.assertLess(y_max, 0.061)

    def test_mesh_generation_su2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            su2_file = Path(tmpdir) / "airfoil.su2"
            generate_naca0012_su2(su2_file, n_foil=32, n_radial=8, r_far=5.0)
            self.assertTrue(su2_file.exists())
            content = su2_file.read_text(encoding="utf-8")
            self.assertIn("NDIME= 2", content)
            self.assertIn("MARKER_TAG= airfoil_wall", content)
            self.assertIn("MARKER_TAG= inlet", content)
            self.assertIn("MARKER_TAG= outlet", content)


if __name__ == "__main__":
    unittest.main()
