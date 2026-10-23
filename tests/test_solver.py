"""Property-oriented checks for the DIAL-Det allocator (CPU, no Torch)."""
import copy
import itertools
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dialdet.certificate import (build_lp, check_certificate, exact_dual_upper,
                                 exact_value, induced_instance)
from dialdet.objective import (GRID_SCALE, admission_cutoff, build_objective, exchange_boundary,
                               isolated_pair_marginal)
from dialdet.solver import SolverConfig, allocate, local_rule


def subsets(n, budget):
    return (chosen for k in range(min(n, budget)+1) for chosen in itertools.combinations(range(n), k))


def random_instance(rng, n):
    q = rng.random(n)
    x = rng.random((n, n))
    r = (x+x.T)/2
    r[r < .3] = 0
    np.fill_diagonal(r, 1.)
    return q, r


class ObjectiveTests(unittest.TestCase):
    def test_marginal_exchange_submodularity_and_grid_error(self):
        rng = np.random.default_rng(1207)
        for n in range(2, 8):
            q, r = random_instance(rng, n)
            obj = build_objective(q, r)
            for chosen in subsets(n, n):
                value = obj.value(chosen)
                eps = (len(chosen)+n+len(chosen)*(len(chosen)-1)//2)/(2*GRID_SCALE)
                self.assertLessEqual(abs(value/GRID_SCALE-obj.float_value(chosen)), eps+1e-14)
                for add in range(n):
                    if add in chosen:
                        continue
                    self.assertEqual(obj.marginal(chosen, add), obj.value(chosen+(add,))-value)
                    if chosen:
                        remove = chosen[0]
                        replaced = tuple(i for i in chosen if i != remove)+(add,)
                        self.assertEqual(obj.exchange(chosen, remove, add), obj.value(replaced)-value)
                        self.assertGreaterEqual(obj.marginal(chosen[1:], add), obj.marginal(chosen, add))

    def test_unthresholded_mass_and_fixed_split_mass(self):
        q = np.array([.8, .7, .6])
        r = np.array([[1., .1, .8], [.1, 1., .1], [.8, .1, 1.]])
        obj = build_objective(q, r)
        np.testing.assert_allclose(obj.mass, (r*r).sum(axis=0))
        self.assertEqual(obj.retained[0, 1], 0)
        self.assertGreater(obj.mass[1], 1)
        sub = obj.subproblem([0])
        self.assertEqual(sub.mass[0], obj.mass[0])
        self.assertLess(sub.b[0, 0]/GRID_SCALE, .2*q[0])

    def test_ideal_block_formula_and_no_duplicates(self):
        labels = np.array([0, 0, 0, 1, 1, 2])
        q = np.array([.92, .85, .6, .8, .4, .7])
        r = (labels[:, None] == labels[None, :]).astype(float)
        obj = build_objective(q, r)
        for chosen in [(0,), (0, 3), (0, 3, 5)]:
            self.assertAlmostEqual(obj.float_value(chosen), 1.2*q[list(chosen)].sum()-.08*len(chosen), places=13)
        out = allocate(q, r, 5)
        self.assertEqual(out["selected"], [0, 3, 5])
        self.assertEqual(out["status"], "rounded_optimum")
        self.assertEqual(len(set(labels[out["selected"]])), len(out["selected"]))

    def test_all_admission_branches_against_direct_marginals(self):
        branches = set()
        for qj in [.01, .0669, .075, .1, .85]:
            threshold = admission_cutoff(.9, qj)
            branches.add(threshold["branch"])
            cut = threshold["affinity_cutoff"]
            for affinity in np.linspace(0, 1, 1001):
                if abs(affinity-cut) > 1e-8:
                    self.assertEqual(isolated_pair_marginal(.9, qj, affinity) > 0, affinity < cut)
        self.assertEqual(branches, {1, 2, 3, 4, 5})

    def test_exchange_closed_form_against_full_objective(self):
        boundaries = []
        for qj in [.1, .5, .7, .847, .9]:
            boundary = exchange_boundary(.9, .85, qj)["affinity_boundary"]
            boundaries.append(boundary)
            for affinity in np.linspace(0, 1, 201):
                r = np.eye(3); r[0, 1] = r[1, 0] = affinity
                obj = build_objective([.9, .85, qj], r)
                difference = obj.float_value([0, 2])-obj.float_value([0, 1])
                if abs(affinity-boundary) > 1e-8:
                    self.assertEqual(difference > 0, affinity > boundary)
        self.assertGreater(boundaries[0], boundaries[1])
        self.assertGreater(boundaries[1], boundaries[2])
        np.testing.assert_allclose(boundaries[:3], [.84, .56, .36], atol=.01)


class AllocatorTests(unittest.TestCase):
    def test_enumeration_matches_global_exhaustion(self):
        rng = np.random.default_rng(9284)
        for n in [0, 1, 2, 5, 7, 9, 12]:
            q, r = random_instance(rng, n)
            obj = build_objective(q, r)
            k = min(4, n)
            out = allocate(q, r, k)
            expected = max(obj.value(chosen) for chosen in subsets(n, k))
            self.assertEqual(out["status"], "rounded_optimum", out.get("fallback_reason"))
            self.assertEqual(out["objective_grid"], expected)
            self.assertTrue(check_certificate(out["certificate"])["valid"])

    def test_forced_lp_bounds_hold_against_exhaustion(self):
        rng = np.random.default_rng(450)
        for n in [3, 5, 7, 8]:
            q, r = random_instance(rng, n)
            config = SolverConfig(exact_limit=1, beam_width=4, max_cut_rounds=3)
            obj = build_objective(q, r, config)
            out = allocate(q, r, 3, config=config)
            self.assertNotEqual(out["status"], "fallback", out.get("fallback_reason"))
            optimum = max(obj.value(chosen) for chosen in subsets(n, 3))
            self.assertGreaterEqual(out["certificate"]["upper_grid"], optimum)
            self.assertLessEqual(out["objective_grid"], optimum)
            self.assertTrue(check_certificate(out["certificate"])["valid"])

    def test_clique_cuts_and_dyadic_dual_residual(self):
        q = np.full(5, .8)
        r = np.ones((5, 5))
        config = SolverConfig(exact_limit=1, max_cut_rounds=3)
        out = allocate(q, r, 3, config=config)
        self.assertNotEqual(out["status"], "fallback", out.get("fallback_reason"))
        self.assertTrue(any(u.get("cuts") for comp in out["certificate"]["components"] for u in comp["upper"]))
        instance = build_objective(q, r).to_record()
        lp = build_lp(instance, 2)
        # Arbitrary, highly infeasible multipliers still yield an upper bound.
        y = np.linspace(-.1, .2, len(lp["rows"]))
        upper, fraction = exact_dual_upper(lp, y, -.13)
        optimum = max(exact_value(instance, chosen) for chosen in itertools.combinations(range(5), 2))
        self.assertGreaterEqual(upper, optimum)
        self.assertGreater(int(fraction["denominator"]), 0)

    def test_split_correction_on_every_report(self):
        rng = np.random.default_rng(1201)
        q, r = random_instance(rng, 7)
        out = allocate(q, r, 4, config=SolverConfig(split_limit=2))
        self.assertNotEqual(out["status"], "fallback", out.get("fallback_reason"))
        cert = out["certificate"]
        obj = build_objective(q, r)
        for chosen in subsets(7, 4):
            split_value = 0
            for comp in cert["components"]:
                chosen_local = [j for j, v in enumerate(comp["members"]) if v in chosen]
                split_value += exact_value(induced_instance(cert["instance"], comp["members"]), chosen_local)
            self.assertLessEqual(obj.value(chosen), split_value+cert["split_correction_grid"])
        optimum = max(obj.value(chosen) for chosen in subsets(7, 4))
        self.assertGreaterEqual(cert["upper_grid"], optimum)
        self.assertEqual(cert["lower_grid"], obj.value(out["selected"]))

    def test_negative_curves_empty_budget_and_ties(self):
        out = allocate(np.zeros(4), np.eye(4), 4)
        self.assertEqual(out["selected"], [])
        self.assertEqual(out["objective_grid"], 0)
        self.assertEqual(out["status"], "rounded_optimum")
        self.assertTrue(any(v is not None and v < 0 for v in out["certificate"]["lower_dp"]["tables"][-1]))
        self.assertEqual(allocate([.9], [[1.]], 0)["selected"], [])
        # Equal-q duplicate identity: ID 'a' wins deterministic component ties.
        tied = allocate([.8, .8], [[1., 1.], [1., 1.]], 1, candidate_ids=["b", "a"])
        self.assertEqual(tied["selected"], [1])
        numeric = allocate([.8, .8], [[1., 1.], [1., 1.]], 1, candidate_ids=[10, 2])
        self.assertEqual(numeric["selected"], [1])

    def test_checker_rejects_tampering(self):
        q, r = random_instance(np.random.default_rng(2), 5)
        out = allocate(q, r, 2, config=SolverConfig(exact_limit=1))
        original = out["certificate"]
        self.assertIsNotNone(original, out.get("fallback_reason"))
        for key in ["lower_grid", "upper_grid", "split_correction_grid", "gap_grid_units"]:
            changed = copy.deepcopy(original)
            changed[key] += 1
            self.assertFalse(check_certificate(changed)["valid"])
        changed = copy.deepcopy(original)
        changed["instance"]["a"][0] += 1
        self.assertFalse(check_certificate(changed)["valid"])
        changed = copy.deepcopy(original)
        proof = next(u for comp in changed["components"] for u in comp["upper"] if u["kind"] == "lp_dual")
        proof["equality_multiplier_hex"] = float(1000).hex()
        self.assertFalse(check_certificate(changed)["valid"])

    def test_missing_lp_multipliers_and_nonfinite_evidence_fallback(self):
        q = np.array([.9, .8, .7])
        r = np.array([[1., .6, .1], [.6, 1., .6], [.1, .6, 1.]])
        failed = SimpleNamespace(ineqlin=SimpleNamespace(marginals=None),
                                 eqlin=SimpleNamespace(marginals=None), message="injected timeout")
        with patch("scipy.optimize.linprog", return_value=failed):
            out = allocate(q, r, 2, config=SolverConfig(exact_limit=1))
        self.assertEqual(out["status"], "fallback")
        self.assertEqual(out["selected"], local_rule(q, r, 2))
        self.assertIsNone(out["gap"])
        self.assertIsNone(out["certificate"])
        bad = q.copy(); bad[0] = float("nan")
        out = allocate(bad, r, 2)
        self.assertEqual(out["status"], "fallback")
        self.assertNotIn(0, out["selected"])


if __name__ == "__main__":
    unittest.main()
