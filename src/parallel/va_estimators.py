"""Modular Va estimators for toy classification and regression.

Swap with --va_method vanilla | parallel_pf | parallel_pp | parallel_hybrid.

vanilla (pF): u ~ p(u|z,D) without x, then H[p(y|x,u,z,D)] from sequential ICL.
parallel_pf: sample (u,y) jointly on one canvas, then evaluate the sampled
y with the same sequential p(y|x,u,z,D) prompt used by vanilla.
parallel_pp: H_P(Y|U) from joint counts (classification) or a fitted joint
Gaussian (regression), with no forward conditional model calls.
The legacy spelling parallel is an alias for parallel_pf.
parallel_hybrid: p(u|x,z,D) comes from the shared canvas, while p(y|x,u,z,D)
comes from the sequential vanilla prompt.  This is retained for old experiments.
"""

from __future__ import annotations

import re
import json
from typing import Protocol

import numpy as np
from src.parallel.parallel_stats import (
    PP_ESTIMATOR,
    canonical_parallel_method,
    joint_count_statistics,
)
from src.parallel.regression_parallel_stats import (
    DEFAULT_PP_COVARIANCE_FLOOR,
    PP_GAUSSIAN_ESTIMATOR,
    joint_gaussian_statistics,
)

from src.chat import (
    U_PLACEHOLDER,
    infer_number_n_masks,
    sample_joint_labels,
    sample_joint_numbers,
    score_joint_label_marginals,
)
from src.utils import (
    GaussianDistribution,
    ToyRegressionUtils,
    calculate_discrete_variance,
    calculate_entropy,
    calculate_kl_divergence,
)


class VaEstimator(Protocol):
    name: str

    def estimate(self, experiment, x_note: str, z_note: str, avg_pyx_probs: dict, Hyx: float, total_variance: float) -> dict:
        ...


class VanillaVaEstimator:
    name = "vanilla"

    def estimate(self, experiment, x_note, z_note, avg_pyx_probs, Hyx, total_variance) -> dict:
        avg_puz_probs = experiment.calculate_avg_probs(z_note, "p(u|z,D)")
        avg_pyxu_z_probs = {}
        for outer_label in experiment.label_keys:
            probability_calculated = f"p(y|x,u={outer_label},z,D)"
            avg_pyxu_z_probs[probability_calculated] = experiment.calculate_avg_probs(
                query_note=x_note,
                probability_calculated=probability_calculated,
                icl_z_note=z_note,
                icl_u_label=outer_label,
            )

        avg_pyxz_probs = {}
        for label in experiment.label_keys:
            avg_pyxz_probs[label] = sum(
                avg_pyxu_z_probs[f"p(y|x,u={u_label},z,D)"][label] * avg_puz_probs[u_label]
                for u_label in experiment.label_keys
            )

        Huz = calculate_entropy(avg_puz_probs)
        Var_uz = calculate_discrete_variance(avg_puz_probs)
        Hyxuz = {f"H[{key}]": calculate_entropy(value) for key, value in avg_pyxu_z_probs.items()}
        Var_yxuz = {f"Var[{key}]": calculate_discrete_variance(value) for key, value in avg_pyxu_z_probs.items()}
        E_Hyxz = 0.0
        E_Var_yxuz = 0.0
        for label in experiment.label_keys:
            E_Hyxz += Hyxuz[f"H[p(y|x,u={label},z,D)]"] * avg_puz_probs[label]
            E_Var_yxuz += Var_yxuz[f"Var[p(y|x,u={label},z,D)]"] * avg_puz_probs[label]
        Va = np.round(E_Hyxz, 5)
        Ve = Hyx - Va
        Va_variance = np.round(E_Var_yxuz, 5)
        Ve_variance = total_variance - Va_variance
        kl_pyx_pyxz = calculate_kl_divergence(avg_pyx_probs, avg_pyxz_probs)
        kl_pyxz_pyx = calculate_kl_divergence(avg_pyxz_probs, avg_pyx_probs)

        save_dict = {}
        for label, prob in avg_puz_probs.items():
            save_dict[f"p(u={label}|z,D)"] = prob
        for key, outer_label_probs in avg_pyxu_z_probs.items():
            for label, prob in outer_label_probs.items():
                new_key = re.sub(r"y", f"y={label}", key, count=1)
                save_dict[new_key] = prob
        for label, prob in avg_pyxz_probs.items():
            save_dict[f"p(y={label}|x,z,D)"] = prob
        save_dict["H[p(u|z,D)]"] = Huz
        save_dict["Var[u|z,D]"] = Var_uz
        for key, entropy in Hyxuz.items():
            save_dict[key] = entropy
        for key, variance in Var_yxuz.items():
            save_dict[key] = variance
        save_dict["Va"] = Va
        save_dict["Ve"] = Ve
        save_dict["Va_variance"] = Va_variance
        save_dict["Ve_variance"] = Ve_variance
        save_dict["kl_pyx_pyxz"] = kl_pyx_pyxz
        save_dict["kl_pyxz_pyx"] = kl_pyxz_pyx
        save_dict["va_method"] = self.name
        return save_dict


def _align_label_probs(probs: dict, label_keys) -> dict:
    aligned = {}
    for key in label_keys:
        if key in probs:
            aligned[key] = float(probs[key])
        elif str(key) in probs:
            aligned[key] = float(probs[str(key)])
        else:
            aligned[key] = 0.0
    total = sum(aligned.values())
    if total > 0:
        aligned = {key: value / total for key, value in aligned.items()}
    return aligned


def _coerce_label_key(value, label_keys):
    """Map a backend label (usually a string) to the dataset's key type."""
    for key in label_keys:
        if value == key or str(value) == str(key):
            return key
    raise ValueError(f"Sampled label {value!r} is not in {list(label_keys)!r}.")


class ParallelPFVaEstimator:
    """MC entropic VUD: joint-canvas samples scored by vanilla's conditional."""

    name = "parallel_pf"

    def __init__(self, num_samples: int):
        if int(num_samples) < 1:
            raise ValueError("parallel classification requires at least 1 joint sample")
        self.num_samples = int(num_samples)

    def estimate(self, experiment, x_note, z_note, avg_pyx_probs, Hyx, total_variance) -> dict:
        label_keys = list(experiment.label_keys)
        joint_counts = {(u, y): 0 for u in label_keys for y in label_keys}
        conditional_sums = {
            u: {y: 0.0 for y in label_keys}
            for u in label_keys
        }
        conditional_counts = {u: 0 for u in label_keys}
        nll_samples: list[float] = []
        conditional_variances: list[float] = []

        attempts = 0
        max_attempts = max(100, 10 * self.num_samples)
        while len(nll_samples) < self.num_samples and attempts < max_attempts:
            attempts += 1
            joint_seed = experiment.num_api_calls
            context_seed = (
                joint_seed
                if experiment.config.permute_context
                else experiment.config.fixed_permutation_seed
            )
            icl = experiment.prompter.note_label_df_to_icl_string(
                experiment.D_note_label_df,
                context_seed,
                z_note,
                U_PLACEHOLDER,
            )
            try:
                sampled_u, sampled_y = sample_joint_labels(
                    icl,
                    x_note,
                    label_keys,
                    seed=joint_seed,
                    temperature=experiment.config.model_temperature,
                    model=experiment.config.model_name,
                )
            except Exception as exc:
                print(f"Joint label sample failed: {type(exc).__name__}: {exc}")
                experiment.num_api_calls += 1
                continue
            experiment.num_api_calls += 1

            calls_before_score = experiment.num_api_calls
            try:
                u = _coerce_label_key(sampled_u, label_keys)
                y = _coerce_label_key(sampled_y, label_keys)
                conditional_probs = experiment.score_query_probs(
                    query_note=x_note,
                    permutation_seed=context_seed,
                    icl_z_note=z_note,
                    icl_u_label=u,
                )
                conditional_probs = _align_label_probs(conditional_probs, label_keys)
                sampled_probability = conditional_probs[y]
                if not np.isfinite(sampled_probability) or sampled_probability <= 0:
                    raise ValueError(
                        f"invalid p(y={y}|x,u={u},z,D)={sampled_probability}"
                    )
            except Exception as exc:
                # score_query_probs increments on success only; still count a failed
                # probability-evaluation attempt so future seeds remain distinct.
                if experiment.num_api_calls == calls_before_score:
                    experiment.num_api_calls += 1
                print(f"Conditional label score failed: {type(exc).__name__}: {exc}")
                continue

            joint_counts[(u, y)] += 1
            conditional_counts[u] += 1
            for label in label_keys:
                conditional_sums[u][label] += conditional_probs[label]
            nll_samples.append(float(-np.log2(sampled_probability)))
            conditional_variances.append(
                float(calculate_discrete_variance(conditional_probs))
            )

        if len(nll_samples) != self.num_samples:
            raise ValueError(
                f"Only {len(nll_samples)}/{self.num_samples} valid joint-label "
                f"samples after {attempts} attempts."
            )

        n = len(nll_samples)
        empirical_joint = {pair: count / n for pair, count in joint_counts.items()}
        empirical_pu = {
            u: sum(empirical_joint[(u, y)] for y in label_keys)
            for u in label_keys
        }
        empirical_py = {
            y: sum(empirical_joint[(u, y)] for u in label_keys)
            for y in label_keys
        }
        avg_conditionals = {
            u: (
                {
                    y: conditional_sums[u][y] / conditional_counts[u]
                    for y in label_keys
                }
                if conditional_counts[u]
                else {y: float("nan") for y in label_keys}
            )
            for u in label_keys
        }

        Va = float(np.round(np.mean(nll_samples), 5))
        Va_variance = float(np.round(np.mean(conditional_variances), 5))

        save_dict = {}
        for u in label_keys:
            save_dict[f"p(u={u}|x,z,D)"] = empirical_pu[u]
            for y in label_keys:
                save_dict[f"pP(y={y},u={u}|x,z,D)"] = empirical_joint[(u, y)]
                save_dict[f"p(y={y}|x,u={u},z,D)"] = avg_conditionals[u][y]
            if conditional_counts[u]:
                save_dict[f"H[p(y|x,u={u},z,D)]"] = calculate_entropy(
                    avg_conditionals[u]
                )
                save_dict[f"Var[p(y|x,u={u},z,D)]"] = (
                    calculate_discrete_variance(avg_conditionals[u])
                )
            else:
                save_dict[f"H[p(y|x,u={u},z,D)]"] = float("nan")
                save_dict[f"Var[p(y|x,u={u},z,D)]"] = float("nan")
        for y in label_keys:
            save_dict[f"p(y={y}|x,z,D)"] = empirical_py[y]
        save_dict["H[p(u|x,z,D)]"] = calculate_entropy(empirical_pu)
        save_dict["Var[u|x,z,D]"] = calculate_discrete_variance(empirical_pu)
        save_dict["Va"] = Va
        save_dict["Ve"] = Hyx - Va
        save_dict["Va_variance"] = Va_variance
        save_dict["Ve_variance"] = total_variance - Va_variance
        save_dict["kl_pyx_pyxz"] = calculate_kl_divergence(
            avg_pyx_probs, empirical_py
        )
        save_dict["kl_pyxz_pyx"] = calculate_kl_divergence(
            empirical_py, avg_pyx_probs
        )
        save_dict["parallel_nll_std"] = float(np.std(nll_samples))
        save_dict["n_joint_samples"] = n
        save_dict["parallel_joint_estimator"] = "mc_joint_sample_conditional_nll_v1"
        save_dict["va_method"] = self.name
        return save_dict


class ParallelPPVaEstimator:
    """Discrete plug-in H_P(Y|U) from shared-canvas samples, without PF calls."""

    name = "parallel_pp"

    def __init__(self, num_samples: int):
        if int(num_samples) < 1:
            raise ValueError("parallel_pp classification requires at least 1 joint sample")
        self.num_samples = int(num_samples)

    def estimate(self, experiment, x_note, z_note, avg_pyx_probs, Hyx, total_variance):
        labels = list(experiment.label_keys)
        counts = {u: {y: 0 for y in labels} for u in labels}
        for _ in range(self.num_samples):
            joint_seed = experiment.num_api_calls
            context_seed = (joint_seed if experiment.config.permute_context
                            else experiment.config.fixed_permutation_seed)
            icl = experiment.prompter.note_label_df_to_icl_string(
                experiment.D_note_label_df, context_seed, z_note, U_PLACEHOLDER,
            )
            # Do not reject/resample outcomes or call the forward scorer.
            try:
                u, y = sample_joint_labels(
                    icl, x_note, labels, seed=joint_seed,
                    temperature=experiment.config.model_temperature,
                    model=experiment.config.model_name,
                )
            finally:
                experiment.num_api_calls += 1
            u, y = _coerce_label_key(u, labels), _coerce_label_key(y, labels)
            counts[u][y] += 1

        stats = joint_count_statistics(counts)
        pu, py = stats["pu"], stats["py"]
        result = {}
        va_variance = 0.0
        for u in labels:
            result[f"p(u={u}|x,z,D)"] = pu[u]
            result[f"n(u={u})"] = stats["u_counts"][u]
            conditional = stats["conditional"][u]
            for y in labels:
                result[f"n(u={u},y={y})"] = counts[u][y]
                result[f"pP(y={y},u={u}|x,z,D)"] = stats["joint"][u][y]
                result[f"p(y={y}|x,u={u},z,D)"] = (
                    conditional[y] if stats["u_counts"][u] else float("nan")
                )
            if stats["u_counts"][u]:
                entropy = calculate_entropy(conditional)
                variance = calculate_discrete_variance(conditional)
                va_variance += pu[u] * variance
            else:
                entropy = variance = float("nan")
            result[f"H[p(y|x,u={u},z,D)]"] = entropy
            result[f"Var[p(y|x,u={u},z,D)]"] = variance
        for y in labels:
            result[f"p(y={y}|x,z,D)"] = py[y]
        result.update({
            "H[p(u|x,z,D)]": calculate_entropy(pu),
            "Var[u|x,z,D]": calculate_discrete_variance(pu),
            "Va": stats["Va"], "Ve": Hyx - stats["Va"],
            "Va_variance": va_variance, "Ve_variance": total_variance - va_variance,
            "kl_pyx_pyxz": calculate_kl_divergence(avg_pyx_probs, py),
            "kl_pyxz_pyx": calculate_kl_divergence(py, avg_pyx_probs),
            "n_joint_samples": stats["n"], "va_method": self.name,
            "parallel_joint_estimator": PP_ESTIMATOR,
        })
        return result


# Backwards-compatible Python import; canonical name and result metadata are PF.
ParallelVaEstimator = ParallelPFVaEstimator


class ParallelHybridVaEstimator:
    """p(u|x,z,D) on the vanilla-aligned canvas; p(y|x,u) via calculate_avg_probs."""

    name = "parallel_hybrid"

    def __init__(self, num_samples: int):
        del num_samples

    def estimate(self, experiment, x_note, z_note, avg_pyx_probs, Hyx, total_variance) -> dict:
        label_keys = list(experiment.label_keys)
        n_perm = max(int(experiment.config.num_permutations), 1)
        puz_acc = {label: 0.0 for label in label_keys}
        py_masked_acc = {label: 0.0 for label in label_keys}

        for _ in range(n_perm):
            permutation_seed = (
                experiment.num_api_calls
                if experiment.config.permute_context
                else experiment.config.fixed_permutation_seed
            )
            icl = experiment.prompter.note_label_df_to_icl_string(
                experiment.D_note_label_df,
                permutation_seed,
                z_note,
                U_PLACEHOLDER,
            )
            p_u, p_y_masked = score_joint_label_marginals(
                icl,
                x_note,
                label_keys,
                temperature=experiment.config.model_temperature,
                model=experiment.config.model_name,
            )
            experiment.num_api_calls += 1
            p_u = _align_label_probs(p_u, label_keys)
            p_y_masked = _align_label_probs(p_y_masked, label_keys)
            for label in label_keys:
                puz_acc[label] += p_u[label]
                py_masked_acc[label] += p_y_masked[label]

        avg_puz_probs = {label: puz_acc[label] / n_perm for label in label_keys}
        avg_py_masked = {label: py_masked_acc[label] / n_perm for label in label_keys}

        avg_pyxu_z_probs = {}
        for outer_label in label_keys:
            probability_calculated = f"p(y|x,u={outer_label},z,D)"
            avg_pyxu_z_probs[outer_label] = experiment.calculate_avg_probs(
                query_note=x_note,
                probability_calculated=probability_calculated,
                icl_z_note=z_note,
                icl_u_label=outer_label,
            )

        Hyxuz = {u: calculate_entropy(avg_pyxu_z_probs[u]) for u in label_keys}
        Var_yxuz = {u: calculate_discrete_variance(avg_pyxu_z_probs[u]) for u in label_keys}
        Va = float(np.round(sum(Hyxuz[u] * avg_puz_probs[u] for u in label_keys), 5))
        Va_variance = float(np.round(sum(Var_yxuz[u] * avg_puz_probs[u] for u in label_keys), 5))
        avg_pyxz_probs = {
            label: sum(avg_pyxu_z_probs[u][label] * avg_puz_probs[u] for u in label_keys)
            for label in label_keys
        }

        save_dict = {}
        for label, prob in avg_puz_probs.items():
            save_dict[f"p(u={label}|x,z,D)"] = prob
        for u in label_keys:
            for label, prob in avg_pyxu_z_probs[u].items():
                save_dict[f"p(y={label}|x,u={u},z,D)"] = prob
            save_dict[f"H[p(y|x,u={u},z,D)]"] = Hyxuz[u]
        for label, prob in avg_pyxz_probs.items():
            save_dict[f"p(y={label}|x,z,D)"] = prob
        for label, prob in avg_py_masked.items():
            save_dict[f"p(y={label}|both_masked)"] = prob
        save_dict["H[p(u|x,z,D)]"] = calculate_entropy(avg_puz_probs)
        save_dict["Va"] = Va
        save_dict["Ve"] = Hyx - Va
        save_dict["Va_variance"] = Va_variance
        save_dict["Ve_variance"] = np.round(total_variance - Va_variance, 5)
        save_dict["kl_pyx_pyxz"] = calculate_kl_divergence(avg_pyx_probs, avg_pyxz_probs)
        save_dict["kl_pyxz_pyx"] = calculate_kl_divergence(avg_pyxz_probs, avg_pyx_probs)
        save_dict["parallel_joint_estimator"] = "hybrid_masked_u_sequential_y_v1"
        save_dict["va_method"] = self.name
        return save_dict


def get_va_estimator(method: str, num_joint_samples: int = 10) -> VaEstimator:
    method = canonical_parallel_method((method or "vanilla").lower())
    if method == "vanilla":
        return VanillaVaEstimator()
    if method == "parallel_pf":
        return ParallelPFVaEstimator(num_joint_samples)
    if method == "parallel_pp":
        return ParallelPPVaEstimator(num_joint_samples)
    if method in {"parallel_hybrid", "hybrid"}:
        return ParallelHybridVaEstimator(num_joint_samples)
    raise ValueError(
        f"Unknown va_method {method!r}. Use vanilla, parallel_pf, parallel_pp, or parallel_hybrid."
    )


class VanillaRegressionVaEstimator:
    name = "vanilla"

    def estimate(self, experiment, x_note, z_note, pyx_gaussian, Hyx, total_variance) -> dict:
        _, u_samples = experiment.calculate_gaussian(z_note, "p(u|z,D)", icl_z_note=z_note)
        pyxuz_distributions: list[GaussianDistribution] = []
        Hyxuz = []
        stds = []
        variances = []
        for u_sample in u_samples:
            pyxuz_gaussian, _ = experiment.calculate_gaussian(
                x_note, "p(y|x,u,z,D)", icl_z_note=z_note, icl_u_label=u_sample
            )
            Hyxuz.append(pyxuz_gaussian.entropy)
            stds.append(pyxuz_gaussian.std)
            variances.append(pyxuz_gaussian.std**2)
            pyxuz_distributions.append(pyxuz_gaussian)

        pyxuz_samples = []
        for _ in range(100):
            u_idx = np.random.randint(len(u_samples))
            pyxuz_samples.append(pyxuz_distributions[u_idx].sample())
        pyxz_gaussian = ToyRegressionUtils.gaussian_from_samples(pyxuz_samples)

        Va = np.round(np.mean(Hyxuz), 5)
        Ve = Hyx - Va
        Va_variance = np.round(np.mean(variances), 5)
        Ve_variance = np.round(total_variance - Va_variance, 5)
        return {
            "p(y|x,z,D)_mean": pyxz_gaussian.mean,
            "p(y|x,z,D)_std": pyxz_gaussian.std,
            "Va": Va,
            "Ve": Ve,
            "Va_variance": Va_variance,
            "Ve_variance": Ve_variance,
            "yxz_std": np.mean(stds),
            "kl_pyx_pyxz": ToyRegressionUtils.calculate_kl_divergence(pyx_gaussian, pyxz_gaussian),
            "kl_pyxz_pyx": ToyRegressionUtils.calculate_kl_divergence(pyxz_gaussian, pyx_gaussian),
            "va_method": self.name,
        }


class ParallelPFRegressionVaEstimator:
    """MC entropic VUD from joint samples scored by vanilla's conditional.

    For each canvas sample (u_i, y_i), fit the same sequential Gaussian
    p(y|x,u_i,z,D) used by vanilla and accumulate -log p(y_i|x,u_i,z,D).
    """

    name = "parallel_pf"

    def __init__(self, num_samples: int, n_masks: int = 8):
        if int(num_samples) < 2:
            raise ValueError("parallel regression requires at least 2 joint samples")
        self.num_samples = int(num_samples)
        self.n_masks_fallback = int(n_masks)

    def estimate(self, experiment, x_note, z_note, pyx_gaussian, Hyx, total_variance) -> dict:
        labels = experiment.D_note_label_df["label"].tolist()
        n_masks = (
            infer_number_n_masks(labels, experiment.config.model_name)
            if labels
            else self.n_masks_fallback
        )
        n_masks = max(int(n_masks), 1)
        print(f"parallel MC-NLL number n_masks={n_masks} from D labels", flush=True)

        joint_samples: list[tuple[float, float]] = []
        conditional_gaussians: list[GaussianDistribution] = []
        nll_samples: list[float] = []
        attempts = 0
        max_attempts = max(100, 10 * self.num_samples)
        while len(joint_samples) < self.num_samples and attempts < max_attempts:
            attempts += 1
            joint_seed = experiment.num_api_calls
            context_seed = (
                joint_seed
                if experiment.config.permute_context
                else experiment.config.fixed_permutation_seed
            )
            icl = experiment.prompter.note_label_df_to_icl_string(
                experiment.D_note_label_df,
                context_seed,
                z_note,
                U_PLACEHOLDER,
            )
            try:
                u, y, _u_ids = sample_joint_numbers(
                    icl,
                    x_note,
                    seed=joint_seed,
                    n_masks=n_masks,
                    temperature=experiment.config.model_temperature,
                    model=experiment.config.model_name,
                )
                if not np.isfinite(u) or not np.isfinite(y):
                    raise ValueError(f"non-finite joint sample {(u, y)}")
            except Exception as exc:
                print(f"Joint number sample failed: {type(exc).__name__}: {exc}")
                experiment.num_api_calls += 1
                continue
            experiment.num_api_calls += 1

            try:
                conditional_gaussian, _ = experiment.calculate_gaussian(
                    x_note,
                    "p(y|x,u,z,D)",
                    icl_z_note=z_note,
                    icl_u_label=float(u),
                )
                conditional_nll = -conditional_gaussian.logpdf(float(y))
            except Exception as exc:
                print(
                    f"Conditional Gaussian evaluation failed for "
                    f"(u={u}, y={y}): {type(exc).__name__}: {exc}"
                )
                continue

            joint_samples.append((float(u), float(y)))
            conditional_gaussians.append(conditional_gaussian)
            nll_samples.append(float(conditional_nll))

        if len(joint_samples) != self.num_samples:
            raise ValueError(
                f"Only {len(joint_samples)}/{self.num_samples} valid joint-number "
                f"samples after {attempts} attempts."
            )

        samples = np.asarray(joint_samples, dtype=float)
        u_samples = samples[:, 0]
        y_samples = samples[:, 1]
        covariance = np.cov(samples, rowvar=False, ddof=1)
        var_u = float(covariance[0, 0])
        var_y = float(covariance[1, 1])
        cov_uy = float(covariance[0, 1])
        marginal_y = ToyRegressionUtils.gaussian_from_samples(y_samples.tolist())
        conditional_variances = [g.std**2 for g in conditional_gaussians]
        conditional_stds = [g.std for g in conditional_gaussians]
        Va = np.round(np.mean(nll_samples), 5)
        Va_variance = np.round(np.mean(conditional_variances), 5)
        correlation = (
            cov_uy / np.sqrt(var_u * var_y)
            if var_u > 0 and var_y > 0
            else 0.0
        )
        correlation = float(np.clip(correlation, -1.0, 1.0))
        return {
            "p(u|x,z,D)_mean": float(np.mean(u_samples)),
            "p(u|x,z,D)_std": float(np.sqrt(max(var_u, 0.0))),
            "p(y|x,z,D)_mean": marginal_y.mean,
            "p(y|x,z,D)_std": marginal_y.std,
            "Cov[u,y|x,z,D]": cov_uy,
            "Corr[u,y|x,z,D]": correlation,
            "Va": Va,
            "Ve": Hyx - Va,
            "Va_variance": Va_variance,
            "Ve_variance": np.round(total_variance - Va_variance, 5),
            "yxz_std": np.mean(conditional_stds),
            "kl_pyx_pyxz": ToyRegressionUtils.calculate_kl_divergence(
                pyx_gaussian, marginal_y
            ),
            "kl_pyxz_pyx": ToyRegressionUtils.calculate_kl_divergence(
                marginal_y, pyx_gaussian
            ),
            "parallel_nll_std": float(np.std(nll_samples)),
            "parallel_joint_estimator": "mc_joint_sample_conditional_nll_v1",
            "va_method": self.name,
            "n_joint_samples": len(joint_samples),
        }


ParallelRegressionVaEstimator = ParallelPFRegressionVaEstimator


class ParallelPPRegressionVaEstimator:
    """Continuous PP via one joint-Gaussian fit, never a PF conditional prompt.

    Fits pooled samples over the context-permutation schedule. Failed numeric
    parses are retried and counted: the resulting distribution is conditional
    on successful finite numeric decoding, as with the existing regression
    samplers. Runtime/model errors propagate rather than being hidden.
    """

    name = "parallel_pp"

    def __init__(self, num_samples: int, n_masks: int = 8,
                 covariance_floor: float = DEFAULT_PP_COVARIANCE_FLOOR):
        if int(num_samples) < 3:
            raise ValueError("Gaussian parallel_pp requires at least 3 joint samples")
        if not np.isfinite(covariance_floor) or covariance_floor < 0:
            raise ValueError("PP covariance floor must be finite and nonnegative")
        self.num_samples = int(num_samples)
        self.n_masks_fallback = int(n_masks)
        self.covariance_floor = float(covariance_floor)

    def estimate(self, experiment, x_note, z_note, pyx_gaussian, Hyx, total_variance):
        if (not np.isfinite([pyx_gaussian.mean, pyx_gaussian.std, Hyx,
                             total_variance]).all() or pyx_gaussian.std <= 0):
            raise ValueError("Parallel PP requires a finite, positive-variance baseline Gaussian")
        labels = experiment.D_note_label_df["label"].tolist()
        n_masks = max(int(infer_number_n_masks(labels, experiment.config.model_name)
                          if labels else self.n_masks_fallback), 1)
        print(f"parallel PP joint-Gaussian n_masks={n_masks}, N={self.num_samples}", flush=True)
        pairs, seeds, context_seeds, failures = [], [], [], []
        max_attempts = max(100, 10 * self.num_samples)
        while len(pairs) < self.num_samples and len(seeds) + len(failures) < max_attempts:
            seed = experiment.num_api_calls
            context_seed = (seed if experiment.config.permute_context
                            else experiment.config.fixed_permutation_seed)
            icl = experiment.prompter.note_label_df_to_icl_string(
                experiment.D_note_label_df, context_seed, z_note, U_PLACEHOLDER,
            )
            try:
                u, y, _ = sample_joint_numbers(
                    icl, x_note, seed=seed, n_masks=n_masks,
                    temperature=experiment.config.model_temperature,
                    model=experiment.config.model_name,
                )
                if not np.isfinite([u, y]).all():
                    raise ValueError(f"non-finite joint sample {(u, y)}")
            except ValueError as exc:
                failures.append({"seed": seed, "context_seed": context_seed, "error": str(exc)})
                print(f"Joint number sample failed: {exc}", flush=True)
                continue
            finally:
                experiment.num_api_calls += 1
            pairs.append((float(u), float(y)))
            seeds.append(seed)
            context_seeds.append(context_seed)
        if len(pairs) != self.num_samples:
            raise ValueError(f"Only {len(pairs)}/{self.num_samples} valid joint-number "
                             f"samples after {max_attempts} attempts")

        stats = joint_gaussian_statistics(pairs, self.covariance_floor)
        var_u, cov_uy = stats["covariance"][0]
        var_y = stats["covariance"][1][1]
        marginal_y = GaussianDistribution(stats["mean"][1], np.sqrt(var_y))
        va = stats["Va"]
        conditional_variance = stats["conditional_variance"]
        if stats["regularized"]:
            print(f"PP covariance regularized: raw eigenvalues={stats['raw_eigenvalues']}, "
                  f"floor={self.covariance_floor}", flush=True)
        return {
            "p(u|x,z,D)_mean": stats["mean"][0],
            "p(u|x,z,D)_std": float(np.sqrt(var_u)),
            "p(y|x,z,D)_mean": marginal_y.mean,
            "p(y|x,z,D)_std": marginal_y.std,
            "Cov[u,y|x,z,D]": cov_uy,
            "Corr[u,y|x,z,D]": stats["correlation"],
            "pP(y|x,u,z,D)_mean_slope": stats["conditional_mean_slope"],
            "pP(y|x,u,z,D)_mean_intercept": stats["conditional_mean_intercept"],
            "Va": va, "Ve": Hyx - va,
            "Va_variance": conditional_variance,
            "Ve_variance": total_variance - conditional_variance,
            "yxz_std": float(np.sqrt(conditional_variance)),
            "kl_pyx_pyxz": ToyRegressionUtils.calculate_kl_divergence(pyx_gaussian, marginal_y),
            "kl_pyxz_pyx": ToyRegressionUtils.calculate_kl_divergence(marginal_y, pyx_gaussian),
            "parallel_joint_estimator": PP_GAUSSIAN_ESTIMATOR,
            "va_method": self.name, "entropy_units": "nats",
            "n_joint_samples": len(pairs), "n_joint_attempts": len(pairs) + len(failures),
            "n_joint_failures": len(failures), "joint_n_masks": n_masks,
            "pp_covariance_floor": self.covariance_floor,
            "pp_covariance_regularized": stats["regularized"],
            "pp_covariance_ddof": 1,
            "joint_samples_json": json.dumps(pairs, allow_nan=False),
            "joint_sample_seeds_json": json.dumps(seeds),
            "joint_context_seeds_json": json.dumps(context_seeds),
            "joint_failures_json": json.dumps(failures),
            "joint_gaussian_fit_json": json.dumps(stats, allow_nan=False),
        }


class ParallelHybridRegressionVaEstimator:
    """u from the vanilla-aligned canvas; H[p(y|x,u)] via calculate_gaussian."""

    name = "parallel_hybrid"

    def __init__(self, num_samples: int, n_masks: int = 8):
        del num_samples
        self.n_masks_fallback = int(n_masks)

    def estimate(self, experiment, x_note, z_note, pyx_gaussian, Hyx, total_variance) -> dict:
        n_draw = (
            int(experiment.config.num_permutations)
            + 2 * int(experiment.config.num_outlier_pairs_to_remove)
        )
        labels = experiment.D_note_label_df["label"].tolist()
        n_masks = (
            infer_number_n_masks(labels, experiment.config.model_name)
            if labels
            else self.n_masks_fallback
        )
        n_masks = max(int(n_masks), 1)
        print(f"parallel number n_masks={n_masks} from D labels", flush=True)

        u_samples = []
        attempts = 0
        while len(u_samples) < n_draw and attempts < 100:
            attempts += 1
            permutation_seed = (
                experiment.num_api_calls
                if experiment.config.permute_context
                else experiment.config.fixed_permutation_seed
            )
            icl = experiment.prompter.note_label_df_to_icl_string(
                experiment.D_note_label_df,
                permutation_seed,
                z_note,
                U_PLACEHOLDER,
            )
            try:
                u, _y, _u_ids = sample_joint_numbers(
                    icl,
                    x_note,
                    seed=experiment.num_api_calls,
                    n_masks=n_masks,
                    temperature=experiment.config.model_temperature,
                    model=experiment.config.model_name,
                )
                u_samples.append(float(u))
            except Exception as exc:
                print(f"Joint number sample failed: {type(exc).__name__}: {exc}")
            experiment.num_api_calls += 1

        if not u_samples:
            raise ValueError("All joint number samples failed for parallel Va.")

        pyxuz_distributions: list[GaussianDistribution] = []
        Hyxuz = []
        stds = []
        variances = []
        for u_sample in u_samples:
            pyxuz_gaussian, _ = experiment.calculate_gaussian(
                x_note, "p(y|x,u,z,D)", icl_z_note=z_note, icl_u_label=u_sample
            )
            Hyxuz.append(pyxuz_gaussian.entropy)
            stds.append(pyxuz_gaussian.std)
            variances.append(pyxuz_gaussian.std**2)
            pyxuz_distributions.append(pyxuz_gaussian)

        pyxuz_samples = []
        for _ in range(100):
            u_idx = np.random.randint(len(u_samples))
            pyxuz_samples.append(pyxuz_distributions[u_idx].sample())
        pyxz_gaussian = ToyRegressionUtils.gaussian_from_samples(pyxuz_samples)

        Va = np.round(np.mean(Hyxuz), 5)
        Va_variance = np.round(np.mean(variances), 5)
        return {
            "p(y|x,z,D)_mean": pyxz_gaussian.mean,
            "p(y|x,z,D)_std": pyxz_gaussian.std,
            "Va": Va,
            "Ve": Hyx - Va,
            "Va_variance": Va_variance,
            "Ve_variance": np.round(total_variance - Va_variance, 5),
            "yxz_std": np.mean(stds),
            "kl_pyx_pyxz": ToyRegressionUtils.calculate_kl_divergence(pyx_gaussian, pyxz_gaussian),
            "kl_pyxz_pyx": ToyRegressionUtils.calculate_kl_divergence(pyxz_gaussian, pyx_gaussian),
            "parallel_joint_estimator": "hybrid_masked_u_sequential_y_v1",
            "va_method": self.name,
            "n_joint_samples": len(u_samples),
        }


def get_regression_va_estimator(
    method: str,
    num_joint_samples: int = 10,
    n_masks: int = 8,
    pp_covariance_floor: float = DEFAULT_PP_COVARIANCE_FLOOR,
):
    method = canonical_parallel_method((method or "vanilla").lower())
    if method == "vanilla":
        return VanillaRegressionVaEstimator()
    if method == "parallel_pf":
        return ParallelPFRegressionVaEstimator(num_joint_samples, n_masks)
    if method == "parallel_pp":
        return ParallelPPRegressionVaEstimator(num_joint_samples, n_masks, pp_covariance_floor)
    if method in {"parallel_hybrid", "hybrid"}:
        return ParallelHybridRegressionVaEstimator(num_joint_samples, n_masks)
    raise ValueError(
        f"Unknown va_method {method!r}. Use vanilla, parallel_pf, parallel_pp, or parallel_hybrid."
    )
