"""Checked DIAL-Det allocation (Algorithm D1), with a transparent reference backend.

This CPU reference backend enumerates small components, searches all cardinalities
with a beam otherwise, uses HiGHS for proposed LP multipliers, and independently
replays the certificate. An LP solver success flag is not an optimality proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
import time

import numpy as np

from .objective import GRID_SCALE, GridObjective, build_objective, canonical_hash
from .certificate import (build_lp, certified_status, check_certificate, exact_dual_upper,
                          induced_instance, max_plus)


@dataclass(frozen=True)
class SolverConfig:
    rho: float = .08
    eta: float = .2
    lam: float = 1.25
    p: float = 2.
    theta: float = .05
    exact_limit: int = 12
    beam_width: int = 64
    split_limit: int = 256
    cert_fraction: float = .015
    lp_time_limit: float | None = None
    max_cut_rounds: int = 3
    max_cliques: int = 2048
    fallback_threshold: float = .5

    def __post_init__(self):
        values = (self.rho, self.eta, self.lam, self.p, self.theta,
                  self.cert_fraction, self.fallback_threshold)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Solver parameters must be finite.")
        if self.rho < 0 or self.eta < 0 or self.lam < 1 or self.p < 1 or not 0 <= self.theta <= 1:
            raise ValueError("Invalid objective parameters.")
        if not 0 <= self.exact_limit <= 20 or self.beam_width < 1 or self.split_limit < 1:
            raise ValueError("Invalid search sizes (exact_limit must be between zero and twenty).")
        if self.cert_fraction < 0 or self.max_cut_rounds < 0 or self.max_cliques < 1:
            raise ValueError("Invalid certification/search limits.")
        if not 0 <= self.fallback_threshold <= 1:
            raise ValueError("Fallback affinity threshold must lie in [0,1].")
        if self.lp_time_limit is not None and (not math.isfinite(self.lp_time_limit) or self.lp_time_limit <= 0):
            raise ValueError("LP time limit must be positive when specified.")


def _identity_info(candidate_ids, n):
    raw = list(range(n)) if candidate_ids is None else list(candidate_ids)
    ids = [str(value) for value in raw]
    keys = [(0, int(value)) if isinstance(value, (int, np.integer)) and not isinstance(value, bool)
            else (1, str(value)) for value in raw]
    return raw, ids, keys


def local_rule(q, r, budget, *, candidate_ids=None, threshold=.5):
    """Matched local rule L: decreasing q, retain iff every raw r is < threshold.

    Finite candidates only are eligible during a numerical fallback. A nonfinite
    pair conservatively blocks the lower-ranked candidate; this sanitation is
    recorded by allocate() when it is needed and must enter risk calibration.
    """
    q, r = np.asarray(q, dtype=float), np.asarray(r, dtype=float)
    _, ids, keys = _identity_info(candidate_ids, len(q))
    order = sorted((i for i in range(len(q)) if math.isfinite(q[i])), key=lambda i: (-q[i], keys[i]))
    result = []
    for i in order:
        if len(result) >= budget:
            break
        if all(math.isfinite(r[i, j]) and r[i, j] < threshold for j in result):
            result.append(i)
    return sorted(result)


def greedy_allocate(q, r, budget, candidate_ids=None, config=None, mass=None):
    """Matched greedy G baseline, using the same executed objective/evidence."""
    obj = build_objective(q, r, config, mass)
    _, ids, keys = _identity_info(candidate_ids, obj.n)
    chosen = []
    while len(chosen) < min(budget, obj.n):
        choices = [(obj.marginal(chosen, j), j) for j in range(obj.n) if j not in chosen]
        if not choices:
            break
        gain, j = min(choices, key=lambda x: (-x[0], keys[x[1]]))
        if gain <= 0:
            break
        chosen.append(j)
    return {"selected": sorted(chosen), "objective": obj.float_value(chosen),
            "objective_grid": obj.value(chosen), "status": "greedy_baseline"}


def _components(obj, ids, split_limit):
    """Deterministic strong-first size-constrained Kruskal partition.

    The paper does not specify an exact weak-edge splitting algorithm. We sort
    retained edges from strongest to weakest and merge components only when the
    resulting size is <= split_limit. All intra-partition edges are then kept;
    all cross-partition interactions enter the recorded split correction.
    """
    n = obj.n
    parent, sizes = list(range(n)), [1]*n

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = [(i, j) for i in range(n) for j in range(i+1, n) if obj.retained[i, j] > 0]
    edges.sort(key=lambda e: (-obj.retained[e], min(ids[e[0]], ids[e[1]]), max(ids[e[0]], ids[e[1]])))
    for i, j in edges:
        a, b = find(i), find(j)
        if a != b and sizes[a]+sizes[b] <= split_limit:
            if a > b:
                a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    result = [sorted(group, key=lambda i: ids[i]) for group in groups.values()]
    result.sort(key=lambda group: tuple(ids[i] for i in group))
    return result


def _enumerate_curve(obj, budget):
    records = []
    for k in range(min(obj.n, budget)+1):
        best, value = None, None
        for chosen in itertools.combinations(range(obj.n), k):
            candidate = obj.value(chosen)
            if value is None or candidate > value:
                best, value = chosen, candidate
        records.append({"selected_local": list(best), "value_grid": int(value)})
    return records


def _first_improvement(obj, chosen):
    selected = tuple(sorted(chosen))
    while selected:
        changed = False
        outside = [j for j in range(obj.n) if j not in selected]
        for remove in selected:
            base = tuple(i for i in selected if i != remove)
            idx = np.asarray(base, dtype=np.int64)
            covered = obj.b[idx].max(axis=0) if base else np.zeros(obj.n, dtype=np.int64)
            gains = obj.a + np.maximum(obj.b-covered[None, :], 0).sum(axis=1, dtype=np.int64)
            if base:
                gains = gains-obj.c[:, idx].sum(axis=1, dtype=np.int64)
            add = next((j for j in outside if gains[j] > gains[remove]), None)
            if add is not None:
                selected = tuple(sorted(base+(add,)))
                changed = True
            if changed:
                break
        if not changed:
            break
    return selected, obj.value(selected)


def _beam_curve(obj, budget, width):
    records = [{"selected_local": [], "value_grid": 0}]
    beam = [((), 0)]
    for k in range(1, min(obj.n, budget)+1):
        expanded = {}
        for chosen, value in beam:
            idx = np.asarray(chosen, dtype=np.int64)
            covered = obj.b[idx].max(axis=0) if chosen else np.zeros(obj.n, dtype=np.int64)
            gains = obj.a + np.maximum(obj.b-covered[None, :], 0).sum(axis=1, dtype=np.int64)
            if chosen:
                gains = gains - obj.c[:, idx].sum(axis=1, dtype=np.int64)
            for j in range(obj.n):
                if j not in chosen:
                    new = tuple(sorted(chosen+(j,)))
                    expanded[new] = value + int(gains[j])
        beam = sorted(expanded.items(), key=lambda item: (-item[1], item[0]))[:width]
        chosen, value = _first_improvement(obj, beam[0][0])
        records.append({"selected_local": list(chosen), "value_grid": int(value)})
    return records


def _maximal_cliques(adjacency, limit):
    """Bounded deterministic Bron-Kerbosch; omitted cuts only loosen the LP."""
    found, visits = [], 0
    truncated = False

    def recurse(current, possible, excluded):
        nonlocal visits, truncated
        visits += 1
        if len(found) >= limit or visits > max(10000, limit*100):
            truncated = True
            return
        if not possible and not excluded:
            if len(current) >= 3:
                found.append(tuple(sorted(current)))
            return
        union = possible | excluded
        pivot = min(union, key=lambda v: (-len(possible & adjacency[v]), v)) if union else None
        candidates = sorted(possible - (adjacency[pivot] if pivot is not None else set()))
        for v in candidates:
            recurse(current+[v], possible & adjacency[v], excluded & adjacency[v])
            possible.remove(v)
            excluded.add(v)
            if truncated:
                return
    recurse([], set(range(len(adjacency))), set())
    return found, truncated


def _lp_upper(obj, instance, cardinality, config):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ArithmeticError("SciPy/HiGHS is required for a non-enumerated component.") from exc
    adjacency = [set(map(int, np.flatnonzero((obj.retained[i] >= .5) & (obj.c[i] > 0)))) - {i} for i in range(obj.n)]
    cliques, truncated = _maximal_cliques(adjacency, config.max_cliques)
    cuts, seen = [], set()
    last_result, lp = None, None
    rounds = 0
    for round_index in range(config.max_cut_rounds+1):
        lp = build_lp(instance, cardinality, cuts)
        rr, cc, vv = [], [], []
        for rownum, terms in enumerate(lp["rows"]):
            for col, value in terms:
                rr.append(rownum); cc.append(col); vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(lp["rows"]), len(lp["coeff"]))).tocsr()
        eq = coo_matrix((np.ones(obj.n), (np.zeros(obj.n, dtype=int), np.arange(obj.n))),
                        shape=(1, len(lp["coeff"]))).tocsr()
        options = {} if config.lp_time_limit is None else {"time_limit": config.lp_time_limit}
        result = linprog(-np.asarray(lp["coeff"], dtype=float)/GRID_SCALE,
                         A_ub=matrix, b_ub=np.asarray(lp["rhs"], dtype=float),
                         A_eq=eq, b_eq=[cardinality], bounds=(0., 1.),
                         method="highs", options=options)
        # A status flag alone certifies nothing. Returned finite multipliers suffice
        # for the residual-corrected upper, even if the solve did not converge.
        y = getattr(getattr(result, "ineqlin", None), "marginals", None)
        beta = getattr(getattr(result, "eqlin", None), "marginals", None)
        if y is None or beta is None or len(beta) != 1:
            raise ArithmeticError(f"LP returned no multipliers at cardinality {cardinality}: {result.message}")
        if not np.isfinite(y).all() or not np.isfinite(beta).all():
            raise ArithmeticError("LP returned non-finite multipliers.")
        last_result = result
        if round_index == config.max_cut_rounds or result.x is None or not np.isfinite(result.x).all():
            break
        additions = []
        for clique in cliques:
            total_x = sum(float(result.x[i]) for i in clique)
            k = int(math.floor(total_x))
            key = (clique, k)
            if total_x <= 1 or k == 0 or key in seen:
                continue
            zsum = sum(float(result.x[lp["z_index"][i, j]]) for i, j in itertools.combinations(clique, 2))
            if k*total_x-k*(k+1)/2-zsum > 1e-8:
                additions.append({"members": list(clique), "k": k})
                seen.add(key)
        if not additions:
            break
        cuts.extend(additions)
        rounds += 1
    y = np.maximum(-np.asarray(last_result.ineqlin.marginals, dtype=float), 0.)
    beta = -float(last_result.eqlin.marginals[0])
    upper, exact_fraction = exact_dual_upper(lp, y, beta)
    return {"kind": "lp_dual", "value_grid": upper,
            "lp_data_hash": lp["data_hash"], "cuts": cuts,
            "inequality_multipliers_hex": [float(v).hex() for v in y],
            "equality_multiplier_hex": beta.hex(), "exact_dual_grid": exact_fraction,
            "solver_status": int(last_result.status), "solver_message": str(last_result.message),
            "cut_rounds": rounds, "clique_search_truncated": truncated}


def allocate(q, r, budget, candidate_ids=None, config=None, mass=None,
             image_id=None, candidate_export_hash=None) -> dict:
    """Run the complete checked allocator and return a JSON-serializable record.

    The caller must calibrate this complete function, including fallback outputs.
    An uncertified solution is retained. Numerical integrity failures use L.
    No native score or bounding box is changed by this function.
    """
    start = time.perf_counter()
    config = SolverConfig() if config is None else (SolverConfig(**config) if isinstance(config, dict) else config)
    q, r = np.asarray(q, dtype=np.float64), np.asarray(r, dtype=np.float64)
    if q.ndim != 1 or r.shape != (len(q), len(q)):
        raise ValueError("Expected q [N], r [N,N].")
    if isinstance(budget, bool) or int(budget) != budget or budget < 0:
        raise ValueError("Budget must be a nonnegative integer.")
    requested_budget, budget = int(budget), min(int(budget), len(q))
    raw_ids, ids, identity_keys = _identity_info(candidate_ids, len(q))
    if len(ids) != len(q) or len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique and aligned with q.")
    timings, obj = {}, None
    try:
        obj = build_objective(q, r, config, mass)
        instance = obj.to_record()
        groups = _components(obj, identity_keys, config.split_limit)
        t = time.perf_counter()
        timings["objective_and_partition_seconds"] = t-start
        records = []
        for members in groups:
            sub = obj.subproblem(members)
            sub_instance = induced_instance(instance, members)
            if len(members) <= config.exact_limit:
                lower = _enumerate_curve(sub, budget)
                upper = [{"kind": "enumeration", "value_grid": x["value_grid"]} for x in lower]
                method = "enumeration"
            else:
                lower = _beam_curve(sub, budget, config.beam_width)
                upper = [{"kind": "zero", "value_grid": 0}]
                for k in range(1, len(lower)):
                    bound = _lp_upper(sub, sub_instance, k, config)
                    if bound["value_grid"] < lower[k]["value_grid"]:
                        raise ArithmeticError("LP residual bound is below a feasible incumbent.")
                    upper.append(bound)
                method = "beam_lp"
            records.append({"members": members, "method": method, "lower": lower, "upper": upper})
        timings["component_search_and_bounds_seconds"] = time.perf_counter()-t
        lower_dp = max_plus([[x["value_grid"] for x in comp["lower"]] for comp in records], budget)
        upper_dp = max_plus([[x["value_grid"] for x in comp["upper"]] for comp in records], budget)
        split_incumbent = sorted(i for comp, k in zip(records, lower_dp["sizes"])
                                 for i in [comp["members"][j] for j in comp["lower"][k]["selected_local"]])
        selected = [] if obj.value(split_incumbent) < 0 else split_incumbent
        part = np.zeros(obj.n, dtype=int)
        for group_index, members in enumerate(groups):
            part[members] = group_index
        cross = part[:, None] != part[None, :]
        lost = np.where(cross, obj.b, 0)
        correction = int(lost.max(axis=0).sum(dtype=np.int64)) if obj.n else 0
        cut_pairs = {(min(i, u), max(i, u)) for i, u, value in instance["b"] if part[i] != part[u]}
        cut_pairs.update((i, j) for i, j, value in instance["c"] if part[i] != part[j])
        lower, upper = obj.value(selected), upper_dp["value"]+correction
        epsilon_twice_units = budget+obj.n+budget*(budget-1)//2
        gap_units = upper-lower+epsilon_twice_units
        status = certified_status(lower, upper, epsilon_twice_units, config.cert_fraction)
        configuration = asdict(config)
        certificate = {
            "schema": "dialdet-grid-certificate-v1", "image_id": None if image_id is None else str(image_id),
            "candidate_export_hash": candidate_export_hash, "candidate_ids": ids,
            "requested_budget": requested_budget, "budget": budget,
            "configuration": configuration, "configuration_hash": canonical_hash(configuration),
            "instance": instance, "instance_hash": canonical_hash(instance),
            "components": records, "coefficient_cut_pairs": [list(p) for p in sorted(cut_pairs)],
            "split_correction_grid": correction, "lower_dp": lower_dp, "upper_dp": upper_dp,
            "split_incumbent": split_incumbent, "selected": selected,
            "lower_grid": lower, "upper_grid": upper,
            "epsilon_twice_grid_units": epsilon_twice_units, "gap_grid_units": gap_units,
            "status": status,
            "arithmetic": "nearest-even binary64 coefficients; exact integer lower and DP; exact dyadic dual residual, ceil to grid",
            "split_rule": "strong-first Kruskal with size-constrained unions",
            "mass_origin": obj.mass_origin,
            "bound_scope": "supplied rounded coefficients; evidence/calibration/labels are not certified by this record",
        }
        t = time.perf_counter()
        checked = check_certificate(certificate)
        timings["independent_check_seconds"] = time.perf_counter()-t
        if not checked["valid"]:
            raise ArithmeticError("Independent checker rejected the certificate: "+"; ".join(checked["errors"]))
        timings["total_seconds"] = time.perf_counter()-start
        return {"selected": selected, "selected_ids": [ids[i] for i in selected],
                "objective": obj.float_value(selected), "objective_grid": lower,
                "upper_bound": upper/GRID_SCALE+epsilon_twice_units/(2*GRID_SCALE),
                "gap": gap_units/GRID_SCALE, "status": status,
                "certificate": certificate, "certificate_check": checked,
                "timings": timings, "mass_origin": obj.mass_origin}
    except (ArithmeticError, ValueError, OverflowError) as exc:
        selected = local_rule(q, r, budget, candidate_ids=raw_ids, threshold=config.fallback_threshold)
        timings["total_seconds"] = time.perf_counter()-start
        return {"selected": selected, "selected_ids": [ids[i] for i in selected],
                "objective": obj.float_value(selected) if obj is not None else None,
                "objective_grid": obj.value(selected) if obj is not None else None,
                "upper_bound": None, "gap": None, "status": "fallback", "certificate": None,
                "certificate_check": {"valid": False, "errors": [str(exc)]},
                "fallback_reason": str(exc), "fallback_threshold": config.fallback_threshold,
                "timings": timings}
