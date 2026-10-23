"""Independent replay of DIAL-Det grid instances and LP dual certificates.

HiGHS proposes multipliers only.  Exact dyadic arithmetic evaluates (C.5),
then rounds each upper bound UP to an integer grid unit.  Integer max-plus
combination consequently needs no floating-point rounding assumptions.
"""
from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

from .objective import GRID_BITS, GRID_SCALE, canonical_hash


def decode_instance(record: dict):
    def integer(value):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("Grid records must contain integers, not rounded or coerced values.")
        return int(value)
    n = integer(record["n"])
    if n < 0 or record["grid_bits"] != GRID_BITS or len(record["a"]) != n:
        raise ValueError("Invalid grid instance dimensions or grid scale.")
    a = [integer(x) for x in record["a"]]
    b, c = {}, {}
    for i, u, value in record["b"]:
        i, u, value = integer(i), integer(u), integer(value)
        if not 0 <= i < n or not 0 <= u < n or value <= 0 or (i, u) in b:
            raise ValueError("Invalid or duplicate coverage coefficient.")
        b[i, u] = value
    for i, j, value in record["c"]:
        i, j, value = integer(i), integer(j), integer(value)
        if not 0 <= i < j < n or value <= 0 or (i, j) in c:
            raise ValueError("Invalid or duplicate pair coefficient.")
        c[i, j] = value
    if sum(abs(x) for x in a) + sum(b.values()) + 2*sum(c.values()) >= 2**53:
        raise ValueError("Grid coefficient integrity limit exceeded.")
    return n, a, b, c


def exact_value(record: dict, selected: Sequence[int]) -> int:
    n, a, b, c = decode_instance(record)
    chosen = set(int(i) for i in selected)
    if len(chosen) != len(selected) or any(i < 0 or i >= n for i in chosen):
        raise ValueError("Invalid selected set.")
    maxima = [0]*n
    for (i, u), value in b.items():
        if i in chosen:
            maxima[u] = max(maxima[u], value)
    return sum(a[i] for i in chosen) + sum(maxima) - sum(
        value for (i, j), value in c.items() if i in chosen and j in chosen)


def induced_instance(record: dict, members: Sequence[int]) -> dict:
    mapping = {int(v): i for i, v in enumerate(members)}
    return {
        "n": len(members), "grid_bits": GRID_BITS,
        "a": [record["a"][int(v)] for v in members],
        "b": sorted([[mapping[i], mapping[u], value] for i, u, value in record["b"]
                     if i in mapping and u in mapping]),
        "c": sorted([[min(mapping[i], mapping[j]), max(mapping[i], mapping[j]), value]
                     for i, j, value in record["c"] if i in mapping and j in mapping]),
    }


def build_lp(record: dict, cardinality: int, cuts=()):
    """Deterministic sparse Equation (D.1) relaxation in max-objective form.

    Row coefficients and RHS are integers; explicit variable bounds are [0,1].
    Only positive b/c variables are needed. Clique cuts use available z pairs.
    """
    n, a, b, c = decode_instance(record)
    pairs_b = sorted(b)
    pairs_c = sorted(c)
    y_index = {pair: n+j for j, pair in enumerate(pairs_b)}
    z_index = {pair: n+len(pairs_b)+j for j, pair in enumerate(pairs_c)}
    coeff = a + [b[pair] for pair in pairs_b] + [-c[pair] for pair in pairs_c]
    rows, rhs = [], []
    for (i, u), col in y_index.items():
        rows.append([(i, -1), (col, 1)])
        rhs.append(0)
    for u in range(n):
        terms = [(col, 1) for (i, unit), col in y_index.items() if unit == u]
        if terms:
            rows.append(terms)
            rhs.append(1)
    for (i, j), col in z_index.items():
        rows += [[(i, 1), (j, 1), (col, -1)], [(i, -1), (col, 1)], [(j, -1), (col, 1)]]
        rhs += [1, 0, 0]
    for cut in cuts:
        members, k = list(map(int, cut["members"])), int(cut["k"])
        if len(set(members)) != len(members) or any(i < 0 or i >= n for i in members) or k < 0:
            raise ValueError("Invalid Boolean-quadric clique cut.")
        terms = [(i, k) for i in members]
        for i, j in itertools.combinations(sorted(members), 2):
            if (i, j) not in z_index:
                raise ValueError("A clique cut references a missing pair variable.")
            terms.append((z_index[i, j], -1))
        rows.append(terms)
        rhs.append(k*(k+1)//2)
    payload = {"instance": record, "cardinality": int(cardinality), "cuts": list(cuts)}
    return {"n": n, "coeff": coeff, "rows": rows, "rhs": rhs,
            "cardinality": int(cardinality), "z_index": z_index,
            "data_hash": canonical_hash(payload)}


def exact_dual_upper(lp, inequality_multipliers, equality_multiplier):
    """Compute the exact dyadic dual residual upper bound, then ceil to grid.

    Multipliers are in physical objective units (as returned by HiGHS).
    Negative inequality multipliers are clipped to zero, as Appendix C.7.
    This remains valid even when the multipliers are not dual feasible.
    """
    if len(inequality_multipliers) != len(lp["rows"]):
        raise ValueError("Wrong number of inequality multipliers.")
    raw_values = [float(v) for v in inequality_multipliers] + [float(equality_multiplier)]
    if not all(math.isfinite(v) for v in raw_values):
        raise ValueError("Non-finite dual multiplier.")
    values = [max(0., v) for v in raw_values[:-1]] + [raw_values[-1]]
    ratios = [v.as_integer_ratio() for v in values]
    denominator = max((den for _, den in ratios), default=1)
    # IEEE binary64 denominators are powers of two, so max is their LCM.
    numerators = [num * (denominator//den) * GRID_SCALE for num, den in ratios]
    y, beta = numerators[:-1], numerators[-1]
    t = [beta if j < lp["n"] else 0 for j in range(len(lp["coeff"]))]
    numerator = beta * lp["cardinality"]
    for row, rhs, multiplier in zip(lp["rows"], lp["rhs"], y):
        numerator += int(rhs)*multiplier
        if multiplier:
            for col, value in row:
                t[col] += int(value)*multiplier
    numerator += sum(max(int(coef)*denominator - total, 0) for coef, total in zip(lp["coeff"], t))
    upper_grid = -((-numerator)//denominator)
    return int(upper_grid), {"numerator": str(numerator), "denominator": str(denominator)}


def certified_status(lower, upper, epsilon_twice_units, fraction):
    """Classify the gap gate with exact integer arithmetic, including its floor."""
    if upper == lower:
        return "rounded_optimum"
    if not math.isfinite(float(fraction)) or fraction < 0:
        raise ValueError("Invalid certification fraction.")
    numerator, denominator = float(fraction).as_integer_ratio()
    gap_units = upper-lower+epsilon_twice_units
    scale_floor = max(2*upper+epsilon_twice_units, 2*GRID_SCALE)
    return "certified" if 2*gap_units*denominator <= numerator*scale_floor else "uncertified"


def max_plus(curves, budget):
    """Exact max-plus DP, preserving infeasibility as None, not as zero."""
    values = [0] + [None]*budget
    tables = [values.copy()]
    choices = []
    for curve in curves:
        nxt, pick = [None]*(budget+1), [None]*(budget+1)
        for total in range(budget+1):
            for k in range(min(total, len(curve)-1)+1):
                old, value = values[total-k], curve[k]
                if old is None or value is None:
                    continue
                candidate = old + value
                if nxt[total] is None or candidate > nxt[total]:
                    nxt[total], pick[total] = candidate, k
        values = nxt
        tables.append(values.copy())
        choices.append(pick)
    feasible = [b for b, val in enumerate(values) if val is not None]
    total = max(feasible, key=lambda b: (values[b], -b))
    optimum, sizes = int(values[total]), []
    remaining = total
    for pick in reversed(choices):
        k = pick[remaining]
        sizes.append(int(k))
        remaining -= k
    sizes.reverse()
    return {"value": optimum, "budget_used": total, "sizes": sizes,
            "tables": tables, "choices": choices}


def check_certificate(record: dict) -> dict:
    """Verify an exported certificate without encoder weights or the LP solver.

    Enumeration records are exhaustively replayed, LP rows are reconstructed,
    multipliers are evaluated with Python integers, both DPs are replayed,
    and the unsplit incumbent and split correction are independently checked.
    """
    try:
        if record.get("schema") != "dialdet-grid-certificate-v1":
            raise ValueError("Unsupported certificate schema.")
        instance = record["instance"]
        n, a, b, c = decode_instance(instance)
        if record["instance_hash"] != canonical_hash(instance):
            raise ValueError("Instance hash mismatch.")
        if record["configuration_hash"] != canonical_hash(record["configuration"]):
            raise ValueError("Configuration hash mismatch.")
        ids = record["candidate_ids"]
        if len(ids) != n or len(set(ids)) != n:
            raise ValueError("Invalid candidate identity list.")
        budget = int(record["budget"])
        if not 0 <= budget <= n:
            raise ValueError("Invalid effective budget.")
        components = record["components"]
        flat = [int(i) for comp in components for i in comp["members"]]
        if sorted(flat) != list(range(n)):
            raise ValueError("Components do not partition the candidate indices.")
        partition = {int(i): j for j, comp in enumerate(components) for i in comp["members"]}
        lost = [0]*n
        cut_pairs = set()
        for (i, u), value in b.items():
            if partition[i] != partition[u]:
                lost[u] = max(lost[u], value)
                cut_pairs.add(tuple(sorted((i, u))))
        for i, j in c:
            if partition[i] != partition[j]:
                cut_pairs.add((i, j))
        correction = sum(lost)
        if int(record["split_correction_grid"]) != correction:
            raise ValueError("Split-coverage correction mismatch.")
        if record["coefficient_cut_pairs"] != [list(p) for p in sorted(cut_pairs)]:
            raise ValueError("Cut-pair ledger mismatch.")
        lower_curves, upper_curves = [], []
        for comp in components:
            sub = induced_instance(instance, comp["members"])
            size = sub["n"]
            needed = min(size, budget)+1
            if len(comp["lower"]) != needed or len(comp["upper"]) != needed:
                raise ValueError("Missing exact-cardinality curve entries.")
            lo, up = [], []
            if comp["method"] == "enumeration":
                if size > int(record["configuration"]["exact_limit"]):
                    raise ValueError("Unbounded enumeration certificate.")
                enumerated = [None]*needed
                for k in range(needed):
                    enumerated[k] = max(exact_value(sub, chosen) for chosen in itertools.combinations(range(size), k))
            else:
                enumerated = None
            for k in range(needed):
                lower = comp["lower"][k]
                selected = list(map(int, lower["selected_local"]))
                if len(selected) != k:
                    raise ValueError("Lower curve cardinality mismatch.")
                value = exact_value(sub, selected)
                if value != int(lower["value_grid"]):
                    raise ValueError("Lower curve value mismatch.")
                lo.append(value)
                upper = comp["upper"][k]
                if enumerated is not None:
                    computed = enumerated[k]
                    if value != computed or upper["kind"] != "enumeration":
                        raise ValueError("Enumeration curve is not exact.")
                elif upper["kind"] == "zero":
                    if k != 0:
                        raise ValueError("A zero upper proof is only valid at k=0.")
                    computed = 0
                else:
                    if upper["kind"] != "lp_dual":
                        raise ValueError("Unsupported upper proof.")
                    lp = build_lp(sub, k, upper["cuts"])
                    if lp["data_hash"] != upper["lp_data_hash"]:
                        raise ValueError("LP data hash mismatch.")
                    multipliers = [float.fromhex(v) for v in upper["inequality_multipliers_hex"]]
                    beta = float.fromhex(upper["equality_multiplier_hex"])
                    computed, fraction = exact_dual_upper(lp, multipliers, beta)
                    if fraction != upper["exact_dual_grid"]:
                        raise ValueError("Exact dual arithmetic mismatch.")
                advertised = int(upper["value_grid"])
                if advertised < computed or advertised < value:
                    raise ValueError("Advertised upper bound is invalid.")
                up.append(advertised)
            lower_curves.append(lo)
            upper_curves.append(up)
        lower_dp = max_plus(lower_curves, budget)
        upper_dp = max_plus(upper_curves, budget)
        if lower_dp != record["lower_dp"] or upper_dp != record["upper_dp"]:
            raise ValueError("Max-plus program mismatch.")
        combined = []
        for comp, k in zip(components, lower_dp["sizes"]):
            combined.extend(comp["members"][i] for i in comp["lower"][k]["selected_local"])
        combined = sorted(combined)
        if combined != record["split_incumbent"]:
            raise ValueError("Split-program incumbent mismatch.")
        selected = sorted(map(int, record["selected"]))
        # Empty-set improvement after split DP is always allowed and recorded.
        expected_selected = [] if exact_value(instance, combined) < 0 else combined
        if selected != expected_selected or len(selected) > budget:
            raise ValueError("Returned incumbent mismatch.")
        lower = exact_value(instance, selected)
        upper = upper_dp["value"] + correction
        if lower != int(record["lower_grid"]) or upper != int(record["upper_grid"]) or upper < lower:
            raise ValueError("Final objective bound mismatch.")
        epsilon_twice_units = budget+n+budget*(budget-1)//2
        gap_units = upper-lower+epsilon_twice_units
        if record["epsilon_twice_grid_units"] != epsilon_twice_units or record["gap_grid_units"] != gap_units:
            raise ValueError("Grid-transfer error mismatch.")
        threshold = float(record["configuration"]["cert_fraction"])
        expected_status = certified_status(lower, upper, epsilon_twice_units, threshold)
        if record["status"] != expected_status:
            raise ValueError("Status disagrees with the checked gap gate.")
        return {"valid": True, "errors": [], "lower_grid": lower, "upper_grid": upper,
                "gap": gap_units/GRID_SCALE, "status": expected_status}
    except (KeyError, TypeError, ValueError, ArithmeticError, OverflowError, IndexError) as exc:
        return {"valid": False, "errors": [str(exc)]}
