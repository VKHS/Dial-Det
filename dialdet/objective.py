"""DIAL-Det's mass-normalized objective, in floating and exact grid arithmetic.

Equations (2), (3), (6), and Appendix C.7 of the DIAL-Det paper.
The mass is formed BEFORE affinity retention and is never changed by a solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

import numpy as np

GRID_BITS = 30
GRID_SCALE = 1 << GRID_BITS
GRID_DELTA = 1.0 / GRID_SCALE


def parameter(config, name, default):
    if config is None:
        return default
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass
class GridObjective:
    q: np.ndarray
    r: np.ndarray
    mass: np.ndarray
    retained: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    parameters: dict
    mass_origin: str = "sum_unthresholded_r_power"

    @property
    def n(self):
        return len(self.q)

    def _indices(self, indices: Iterable[int]):
        result = tuple(int(i) for i in indices)
        if len(set(result)) != len(result) or any(i < 0 or i >= self.n for i in result):
            raise ValueError("Selected indices must be unique and in range.")
        return result

    def value(self, indices: Iterable[int]) -> int:
        """Exact integer value in units of delta; the empty set has value zero."""
        chosen = self._indices(indices)
        if not chosen:
            return 0
        idx = np.asarray(chosen, dtype=np.int64)
        unary = int(self.a[idx].sum(dtype=np.int64))
        coverage = int(self.b[idx, :].max(axis=0).sum(dtype=np.int64))
        penalty = int(np.triu(self.c[np.ix_(idx, idx)], 1).sum(dtype=np.int64))
        return unary + coverage - penalty

    def marginal(self, indices: Iterable[int], candidate: int) -> int:
        """Equation (6), applied to the executed integer-grid coefficients."""
        chosen = self._indices(indices)
        j = int(candidate)
        if j in chosen or not 0 <= j < self.n:
            raise ValueError("Marginal candidate must be outside the selected set.")
        idx = np.asarray(chosen, dtype=np.int64)
        covered = self.b[idx, :].max(axis=0) if chosen else np.zeros(self.n, dtype=np.int64)
        gain = int(np.maximum(self.b[j] - covered, 0).sum(dtype=np.int64))
        return int(self.a[j]) + gain - int(self.c[j, idx].sum(dtype=np.int64))

    def exchange(self, indices: Iterable[int], remove: int, add: int) -> int:
        """Exact exchange gain: add-j marginal minus add-i marginal after removing i."""
        chosen = self._indices(indices)
        if remove not in chosen or add in chosen:
            raise ValueError("An exchange removes a selected index and adds an outside index.")
        base = tuple(i for i in chosen if i != remove)
        return self.marginal(base, add) - self.marginal(base, remove)

    def float_value(self, indices: Iterable[int]) -> float:
        """Unrounded Equation (3), evaluated on the original unsplit evidence."""
        chosen = self._indices(indices)
        if not chosen:
            return 0.0
        idx = np.asarray(chosen, dtype=np.int64)
        p = self.parameters
        coverage = (self.q[idx, None] * self.retained[idx, :]).max(axis=0)
        pair = self.retained[np.ix_(idx, idx)] * np.minimum(self.q[idx, None], self.q[None, idx])
        return float((self.q[idx] - p["rho"]).sum()
                     + p["eta"] * (coverage / self.mass).sum()
                     - p["lam"] * np.triu(pair, 1).sum())

    def subproblem(self, indices: Iterable[int]):
        """An induced split component. Mass-derived coefficients remain fixed."""
        idx = np.asarray(tuple(indices), dtype=np.int64)
        return GridObjective(self.q[idx], self.r[np.ix_(idx, idx)], self.mass[idx],
                             self.retained[np.ix_(idx, idx)], self.a[idx],
                             self.b[np.ix_(idx, idx)], self.c[np.ix_(idx, idx)],
                             self.parameters.copy(), self.mass_origin)

    def to_record(self) -> dict:
        """Sparse, lossless grid instance; all coefficient values are integers."""
        bi, bu = np.nonzero(self.b)
        ci, cj = np.nonzero(np.triu(self.c, 1))
        return {
            "n": self.n, "grid_bits": GRID_BITS,
            "a": self.a.tolist(),
            "b": [[int(i), int(u), int(self.b[i, u])] for i, u in zip(bi, bu)],
            "c": [[int(i), int(j), int(self.c[i, j])] for i, j in zip(ci, cj)],
        }


def build_objective(q, r, config=None, mass=None) -> GridObjective:
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if q.ndim != 1 or r.shape != (q.size, q.size):
        raise ValueError("Expected quality [N] and symmetric affinity [N,N].")
    if not np.isfinite(q).all() or not np.isfinite(r).all():
        raise ValueError("Quality and affinity must be finite.")
    if np.any((q < 0) | (q > 1)) or np.any((r < 0) | (r > 1)):
        raise ValueError("Quality and affinity must lie in [0,1].")
    if not np.allclose(r, r.T, atol=1e-10, rtol=0):
        raise ValueError("Identity affinity must be symmetric.")
    if not np.allclose(np.diag(r), 1, atol=1e-8, rtol=0):
        raise ValueError("The identity affinity diagonal must equal one.")
    # Canonicalize only roundoff-size differences, explicitly recorded in metadata.
    r = (r + r.T) * 0.5
    np.fill_diagonal(r, 1.0)
    p = {name: float(parameter(config, name, default)) for name, default in
         (("rho", .08), ("eta", .2), ("lam", 1.25), ("p", 2.), ("theta", .05))}
    if not all(np.isfinite(v) for v in p.values()):
        raise ValueError("Objective parameters must be finite.")
    if p["rho"] < 0 or p["eta"] < 0 or p["lam"] < 1 or p["p"] < 1 or not 0 <= p["theta"] <= 1:
        raise ValueError("Require rho>=0, eta>=0, lam>=1, p>=1 and 0<=theta<=1.")
    sharpened = np.power(r, p["p"])
    expected_mass = sharpened.sum(axis=0)
    origin = "sum_unthresholded_r_power"
    if mass is None:
        mass = expected_mass
    else:
        mass = np.asarray(mass, dtype=np.float64)
        if mass.shape != q.shape or not np.isfinite(mass).all() or np.any(mass < 1):
            raise ValueError("An external fixed mass must be finite, shape [N], and at least one.")
        origin = "external_fixed_mass" if not np.allclose(mass, expected_mass, atol=1e-8, rtol=1e-6) else origin
    retained = np.where(sharpened >= p["theta"], sharpened, 0.)
    real_a = q - p["rho"]
    real_b = p["eta"] * q[:, None] * retained / mass[None, :]
    real_c = p["lam"] * retained * np.minimum(q[:, None], q[None, :])
    np.fill_diagonal(real_c, 0.)
    # The paper uses exactly represented binary64 sums below 2^53 grid units.
    # This implementation uses int64 sums and enforces the same stricter limit.
    magnitude = float(np.abs(real_a).sum() + real_b.sum() + real_c.sum())
    if not np.isfinite(magnitude) or magnitude * GRID_SCALE + q.size**2 >= 2**53:
        raise ArithmeticError("Coefficient sums exceed the paper's exact-grid integrity limit.")
    a = np.rint(real_a * GRID_SCALE).astype(np.int64)
    b = np.rint(real_b * GRID_SCALE).astype(np.int64)
    c = np.rint(real_c * GRID_SCALE).astype(np.int64)
    return GridObjective(q.copy(), r, mass.copy(), retained, a, b, c, p, origin)


def objective(q, r, selected, config=None, mass=None) -> float:
    return build_objective(q, r, config, mass).float_value(selected)


def isolated_pair_marginal(qi, qj, affinity, *, outside_mass=0., rho=.08,
                           eta=.2, lam=1.25, p=2., theta=.05):
    """Appendix C.1, with its stated isolation/covered-other-units assumptions."""
    if not 0 < qj <= qi <= 1 or not 0 <= affinity <= 1 or outside_mass < 0:
        raise ValueError("Require 0<qj<=qi<=1, affinity in [0,1], outside_mass>=0.")
    x = affinity**p
    z = x if x >= theta else 0.
    return qj - rho - lam*qj*z + eta*max(qj-qi*z, 0.)/(1.+outside_mass+x)


def admission_cutoff(qi, qj, *, outside_mass=0., rho=.08, eta=.2, lam=1.25, p=2., theta=.05):
    """Complete five-branch affinity threshold in Appendix C.2 (strict admission)."""
    if not 0 < qj <= qi <= 1 or outside_mass < 0:
        raise ValueError("Require 0<qj<=qi<=1 and nonnegative outside_mass.")
    a = 1.+outside_mass
    d0 = qj-rho+eta*qj/a
    dm = qj-rho+eta*qj/(a+theta)
    dp = qj-rho-lam*qj*theta+eta*max(qj-qi*theta, 0.)/(a+theta)
    if d0 <= 0:
        x, branch = 0., 1
    elif dm <= 0:
        x, branch = eta*qj/(rho-qj)-a, 2
    elif dp <= 0:
        x, branch = theta, 3
    else:
        t0 = (1-rho/qj)/lam
        if t0 >= qj/qi:
            x, branch = t0, 4
        else:
            A, B, C = lam*qj, lam*qj*a-(qj-rho)+eta*qi, (qj-rho)*a+eta*qj
            root = float(np.sqrt(B*B+4*A*C))
            x = 2*C/(B+root) if B >= 0 else (-B+root)/(2*A)
            branch = 5
    return {"affinity_cutoff": float(np.clip(x, 0., 1.)**(1/p)), "power_cutoff": float(x), "branch": branch}


def exchange_boundary(qi, qi_prime, qj, *, rho=.08, eta=.2, lam=1.25, p=2., theta=.05):
    """Appendix C.3-C.4: K=2 exchange of a near duplicate for an isolated object.

    Assumptions: qi >= qi_prime > rho; qj > rho/(1+eta); no other affinities.
    Above the returned affinity boundary, {i,j} beats {i,i_prime}; equality can
    be a tie or the retention jump and should be evaluated with the objective.
    """
    if not 0 <= rho < qi_prime <= qi <= 1 or not rho/(1+eta) < qj <= 1:
        raise ValueError("The qualities do not satisfy Proposition C2's assumptions.")
    if qj > qi_prime:
        s, branch = 0., 1
    elif qi_prime*(1+eta/(1+theta)) < (1+eta)*qj:
        s, branch = eta/((1+eta)*qj/qi_prime-1)-1, 2
    else:
        A = lam*qi_prime
        B = eta*qi+(1+eta)*qj+(lam-1)*qi_prime
        C = (1+eta)*(qi_prime-qj)
        root = float(np.sqrt(B*B+4*A*C))
        sigma = 2*C/(B+root) if B >= 0 else (-B+root)/(2*A)
        if qi*sigma >= qi_prime:
            sigma = (qi_prime-(1+eta)*qj)/(lam*qi_prime)
        s, branch = max(theta, sigma), 3
    return {"affinity_boundary": float(np.clip(s, 0., 1.)**(1/p)), "power_boundary": float(s), "branch": branch}
