"""Discrete Parallel PP plug-in estimator, using joint counts only."""

import math


PP_ESTIMATOR = "mc_joint_counts_conditional_entropy_v1"
PP_EXACT_ESTIMATOR = "exact_sampler_joint_conditional_entropy_v1"


def canonical_parallel_method(method):
    """Keep the old CLI spelling as a PF alias, never as a PP alias."""
    return "parallel_pf" if method == "parallel" else method


def joint_count_statistics(counts):
    """Estimate H_P(Y|U) in bits without smoothing or conditional prompts.

    Counts are pooled over the sampling schedule's context permutations.
    Unobserved u groups have undefined conditionals (None), not invented
    uniform distributions; they contribute zero to the empirical expectation.
    """
    if not counts or not next(iter(counts.values())):
        raise ValueError("Joint counts must have nonempty label sets.")
    y_keys = list(next(iter(counts.values())))
    for row in counts.values():
        if set(row) != set(y_keys):
            raise ValueError("Joint count rows must share the same y labels.")
        if any(not math.isfinite(c) or c < 0 or int(c) != c for c in row.values()):
            raise ValueError("Joint counts must be finite nonnegative integers.")
    u_counts = {u: sum(row.values()) for u, row in counts.items()}
    n = sum(u_counts.values())
    if n == 0:
        raise ValueError("Parallel PP requires at least one joint sample.")
    joint = {u: {y: c / n for y, c in row.items()} for u, row in counts.items()}
    pu = {u: c / n for u, c in u_counts.items()}
    py = {y: sum(row[y] for row in joint.values()) for y in y_keys}
    conditional = {
        u: {y: c / u_counts[u] if u_counts[u] else None for y, c in row.items()}
        for u, row in counts.items()
    }
    va = -sum(
        c / n * math.log2(c / u_counts[u])
        for u, row in counts.items() for c in row.values() if c > 0
    )
    return {
        "joint": joint, "pu": pu, "py": py, "conditional": conditional,
        "u_counts": u_counts, "n": n, "Va": float(va),
    }


def joint_distribution_statistics(joint):
    """Exact H_P(Y|U) in bits from a joint {u: {y: probability}} (parallel_pp_exact).

    The probability-weighted analogue of joint_count_statistics: the same plug-in formula with
    the sampler's exact law in place of empirical frequencies."""
    if not joint or not next(iter(joint.values())):
        raise ValueError("Joint distribution must have nonempty label sets.")
    y_keys = list(next(iter(joint.values())))
    total = sum(p for row in joint.values() for p in row.values())
    if not all(math.isfinite(p) and p >= 0 for row in joint.values() for p in row.values()) or not total > 0:
        raise ValueError("Joint probabilities must be finite, nonnegative and not all zero.")
    joint = {u: {y: p / total for y, p in row.items()} for u, row in joint.items()}
    pu = {u: sum(row.values()) for u, row in joint.items()}
    py = {y: sum(row[y] for row in joint.values()) for y in y_keys}
    conditional = {
        u: {y: p / pu[u] if pu[u] > 0 else None for y, p in row.items()}
        for u, row in joint.items()
    }
    va = -sum(
        p * math.log2(p / pu[u])
        for u, row in joint.items() for p in row.values() if p > 0
    )
    return {"joint": joint, "pu": pu, "py": py, "conditional": conditional, "Va": float(va)}
