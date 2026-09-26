"""White-box and black-box acceptance tests for the SIH26120 prototype."""

import os
import unittest

import requests

from digital_twin import IntegratedDigitalTwin
from optimizer import SimplifiedOptimizer


BASE_URL = os.environ.get("SIH_BASE_URL", "http://localhost:8001")
BASELINE = {
    "steam_volume": 80,
    "steam_pressure": 25,
    "injection_duration": 24,
    "soak_time": 36,
    "spm": 4,
    "stroke_length": 86,
    "vfd_frequency": 45,
    "production_cutoff": 1,
}


class WhiteBoxTests(unittest.TestCase):
    def test_identical_twin_inputs_are_deterministic(self):
        first = IntegratedDigitalTwin().simulate_css_cycle(**BASELINE)
        second = IntegratedDigitalTwin().simulate_css_cycle(**BASELINE)
        self.assertEqual(first, second)

    def test_css_and_srp_controls_change_outputs(self):
        baseline = IntegratedDigitalTwin().simulate_css_cycle(**BASELINE)
        changed = dict(BASELINE, steam_volume=100, spm=6.5, stroke_length=105)
        proposed = IntegratedDigitalTwin().simulate_css_cycle(**changed)
        self.assertGreater(proposed.current_production, baseline.current_production)
        self.assertGreater(proposed.current_energy, baseline.current_energy)
        self.assertNotEqual(proposed.current_sor, baseline.current_sor)

    def test_optimizer_rejects_unsafe_results(self):
        optimizer = SimplifiedOptimizer()
        solutions = optimizer.optimize(10)
        self.assertTrue(solutions)
        self.assertTrue(all(s["valid"] for s in solutions))
        self.assertTrue(all(s["well_state"].failure_risk <= optimizer.constraints.max_failure_risk for s in solutions))


class BlackBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=3)
        except requests.RequestException as error:
            raise unittest.SkipTest(f"API unavailable at {BASE_URL}: {error}")
        if response.status_code != 200:
            raise unittest.SkipTest(f"API unavailable at {BASE_URL}")

    def test_core_endpoints(self):
        checks = [
            ("get", "/health", None),
            ("get", "/api/models/status", None),
            ("post", "/api/simulate", BASELINE),
            ("post", "/api/predict", BASELINE),
            ("get", "/api/simulate/what-if", dict(BASELINE, steam_volume=100)),
            ("post", "/api/optimize", {"n_iterations": 10, "priority": "balanced"}),
        ]
        for method, path, payload in checks:
            response = getattr(requests, method)(
                BASE_URL + path,
                json=payload if method == "post" else None,
                params=payload if method == "get" and payload else None,
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, path)
            self.assertTrue(response.json().get("success", True), path)

    def test_invalid_input_is_rejected(self):
        response = requests.post(BASE_URL + "/api/simulate", json=dict(BASELINE, spm=99), timeout=10)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)