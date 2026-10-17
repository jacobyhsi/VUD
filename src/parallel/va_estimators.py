"""Modular Va estimators for toy classification and regression.

Swap with --va_method vanilla | backward | parallel_pf | parallel_pp | parallel_pp_exact.

vanilla (pF): u ~ p(u|z,D) without x, then H[p(y|x,u,z,D)] from sequential ICL.
backward (pB): p(y|x,D) p(u|z,y,x,D) from the same prompts with roles swapped; H[pB(y|x,u,z,D)]
by Bayes' rule, with the KL filter on the u-marginal.
parallel_pf: sample (u,y) jointly on one canvas, then evaluate the sampled
y with the same sequential p(y|x,u,z,D) prompt used by vanilla.
parallel_pp: H_P(Y|U) from joint counts (classification) or a fitted joint
Gaussian (regression), with no forward conditional model calls.
parallel_pp_exact: the same H_P(Y|U) from the joint sampler's exact law instead of
counts (small-MDLM classification), with no Monte Carlo error.
The legacy spelling parallel is an alias for parallel_pf.
"""

from __future__ import annotations

import json
from typing import Protocol

import numpy as np
from src.parallel.parallel_stats import (
    PP_ESTIMATOR,
    PP_EXACT_ESTIMATOR,
    joint_distribution_statistics,
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
    sample_joint_labels_batch,
    sample_joint_numbers,
    score_joint_label_law,
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


def _weighted_conditional_fields(label_keys, pu, pyxu_z, Hyx, total_variance, u_given="z,D"):
    """Va = sum_u pu(u) H[p(y|x,u,z,D)] and its companion fields, keyed as vanilla saves them.

    pu are the auxiliary-label weights p(u|<u_given>) and pyxu_z[u] the conditionals
    p(y|x,u,z,D). Returns (fields, p(y|x,z,D)); the caller adds its KL filter."""
    pyxz = {label: sum(pyxu_z[u][label] * pu[u] for u in label_keys) for label in label_keys}
    Hyxuz = {u: calculate_entropy(pyxu_z[u]) for u in label_keys}
    Var_yxuz = {u: calculate_discrete_variance(pyxu_z[u]) for u in label_keys}
    Va = np.round(sum(Hyxuz[u] * pu[u] for u in label_keys), 5)
    Va_variance = np.round(sum(Var_yxuz[u] * pu[u] for u in label_keys), 5)

    fields = {}
    for label, prob in pu.items():
        fields[f"p(u={label}|{u_given})"] = prob
    for u in label_keys:
        for label, prob in pyxu_z[u].items():
            fields[f"p(y={label}|x,u={u},z,D)"] = prob
    for label, prob in pyxz.items():
        fields[f"p(y={label}|x,z,D)"] = prob
    fields[f"H[p(u|{u_given})]"] = calculate_entropy(pu)
    fields[f"Var[u|{u_given}]"] = calculate_discrete_variance(pu)
    for u in label_keys:
        fields[f"H[p(y|x,u={u},z,D)]"] = Hyxuz[u]
    for u in label_keys:
        fields[f"Var[p(y|x,u={u},z,D)]"] = Var_yxuz[u]
    fields["Va"] = Va
    fields["Ve"] = Hyx - Va
    fields["Va_variance"] = Va_variance
    fields["Ve_variance"] = total_variance - Va_variance
    return fields, pyxz


class VanillaVaEstimator:
    """Forward factorization pF(y,u|x,z,D) = p(u|z,D) p(y|x,u,z,D)."""

    name = "vanilla"

    def estimate(self, experiment, x_note, z_note, avg_pyx_probs, Hyx, total_variance) -> dict:
        avg_puz_probs = experiment.calculate_avg_probs(z_note, "p(u|z,D)")
        avg_pyxu_z_probs = {
            u: experiment.calculate_avg_probs(
                query_note=x_note,
                probability_calculated=f"p(y|x,u={u},z,D)",
                icl_z_note=z_note,
                icl_u_label=u,
            )
            for u in experiment.label_keys
        }
        save_dict, avg_pyxz_probs = _weighted_conditional_fields(
            experiment.label_keys, avg_puz_probs, avg_pyxu_z_probs, Hyx, total_variance
        )
        save_dict["kl_pyx_pyxz"] = calculate_kl_divergence(avg_pyx_probs, avg_pyxz_probs)
        save_dict["kl_pyxz_pyx"] = calculate_kl_divergence(avg_pyxz_probs, avg_pyx_probs)
        save_dict["va_method"] = self.name
        return save_dict


class BackwardVaEstimator:
    """Backward factorization pB(y,u|x,z,D) = p(y|x,D) p(u|z,y,x,D).

    The same prompts as vanilla with the roles of (x, y) and (z, u) swapped; p(y|x,u,z,D) then
    comes from the joint by Bayes' rule, and Va = sum_u pB(u|x,z,D) H[pB(y|x,u,z,D)]. pB keeps the
    prompted y-marginal, so the KL filter checks the u-marginal instead: kl_pyx_pyxz holds
    KL(p(u|z,D) || pB(u|x,z,D)) and kl_pyxz_pyx the reverse, so z selection works unchanged."""

    name = "backward"

    def estimate(self, experiment, x_note, z_note, avg_pyx_probs, Hyx, total_variance) -> dict:
        labels = list(experiment.label_keys)
        avg_puz_probs = experiment.calculate_avg_probs(z_note, "p(u|z,D)")
        avg_puzxy_probs = {
            y: experiment.calculate_avg_probs(
                query_note=z_note,
                probability_calculated=f"p(u|z,x,y={y},D)",
                icl_z_note=x_note,
                icl_u_label=y,
            )
            for y in labels
        }
        joint = {u: {y: avg_pyx_probs[y] * avg_puzxy_probs[y][u] for y in labels} for u in labels}
        stats = joint_distribution_statistics(joint)
        # A u with zero backward probability carries zero weight; any conditional will do.
        conditional = {
            u: {y: avg_pyx_probs[y] if p is None else p for y, p in stats["conditional"][u].items()}
            for u in labels
        }
        save_dict, _ = _weighted_conditional_fields(
            labels, stats["pu"], conditional, Hyx, total_variance, u_given="x,z,D"
        )
        for label, prob in avg_puz_probs.items():
            save_dict[f"p(u={label}|z,D)"] = prob
        for y in labels:
            for u in labels:
                save_dict[f"p(u={u}|z,x,y={y},D)"] = avg_puzxy_probs[y][u]
                save_dict[f"pB(y={y},u={u}|x,z,D)"] = stats["joint"][u][y]
        save_dict["kl_pyx_pyxz"] = calculate_kl_divergence(avg_puz_probs, stats["pu"])
        save_dict["kl_pyxz_pyx"] = calculate_kl_divergence(stats["pu"], avg_puz_probs)
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
        if experiment.config.model_name.startswith("mdlm:"):
            # Small MDLMs draw all samples in one batch: sample i keeps its own context permutation.
            first_seed = experiment.num_api_calls
            icls = [
                experiment.prompter.note_label_df_to_icl_string(
                    experiment.D_note_label_df,
                    seed if experiment.config.permute_context else experiment.config.fixed_permutation_seed,
                    z_note, U_PLACEHOLDER,
                )
                for seed in range(first_seed, first_seed + self.num_samples)
            ]
            try:
                samples = sample_joint_labels_batch(
                    icls, x_note, labels, seed=first_seed,
                    temperature=experiment.config.model_temperature,
                    model=experiment.config.model_name,
                )
            finally:
                experiment.num_api_calls += self.num_samples
        else:
            samples = None
        for _ in range(0 if samples is not None else self.num_samples):
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
        for u, y in samples or []:
            counts[_coerce_label_key(u, labels)][_coerce_label_key(y, labels)] += 1

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


class ParallelPPExactVaEstimator:
    """parallel_pp without Monte Carlo: H_P(Y|U) from the sampler's exact joint law.

    The joint is averaged over num_permutations context permutations, as parallel_pp pools its
    samples over permutations. Also reports Va under the u-first and y-first reveal orders alone,
    and their total-variation distance (path dependence on the shared canvas)."""

    name = "parallel_pp_exact"

    def estimate(self, experiment, x_note, z_note, avg_pyx_probs, Hyx, total_variance):
        labels = list(experiment.label_keys)
        n_perm = max(int(experiment.config.num_permutations), 1)
        first_seed = experiment.num_api_calls
        icls = [
            experiment.prompter.note_label_df_to_icl_string(
                experiment.D_note_label_df,
                seed if experiment.config.permute_context else experiment.config.fixed_permutation_seed,
                z_note, U_PLACEHOLDER,
            )
            for seed in range(first_seed, first_seed + n_perm)
        ]
        try:
            laws = score_joint_label_law(
                icls, x_note, labels,
                temperature=experiment.config.model_temperature,
                model=experiment.config.model_name,
            )
        finally:
            experiment.num_api_calls += n_perm
        averaged = {
            part: {u: {y: sum(law[part][(u, y)] for law in laws) / n_perm for y in labels} for u in labels}
            for part in ("joint", "u_first", "y_first")
        }
        stats = joint_distribution_statistics(averaged["joint"])
        orders = {part: joint_distribution_statistics(averaged[part]) for part in ("u_first", "y_first")}
        pu, py = stats["pu"], stats["py"]
        result = {}
        va_variance = 0.0
        for u in labels:
            result[f"p(u={u}|x,z,D)"] = pu[u]
            conditional = stats["conditional"][u]
            for y in labels:
                result[f"pP(y={y},u={u}|x,z,D)"] = stats["joint"][u][y]
                result[f"p(y={y}|x,u={u},z,D)"] = conditional[y] if pu[u] > 0 else float("nan")
            if pu[u] > 0:
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
            "Va_u_first": orders["u_first"]["Va"], "Va_y_first": orders["y_first"]["Va"],
            "path_tv": 0.5 * sum(
                abs(orders["u_first"]["joint"][u][y] - orders["y_first"]["joint"][u][y])
                for u in labels for y in labels
            ),
            "n_permutations": n_perm, "va_method": self.name,
            "parallel_joint_estimator": PP_EXACT_ESTIMATOR,
        })
        return result


# Backwards-compatible Python import; canonical name and result metadata are PF.
ParallelVaEstimator = ParallelPFVaEstimator


def get_va_estimator(method: str, num_joint_samples: int = 10) -> VaEstimator:
    method = canonical_parallel_method((method or "vanilla").lower())
    if method == "vanilla":
        return VanillaVaEstimator()
    if method == "backward":
        return BackwardVaEstimator()
    if method == "parallel_pf":
        return ParallelPFVaEstimator(num_joint_samples)
    if method == "parallel_pp":
        return ParallelPPVaEstimator(num_joint_samples)
    if method == "parallel_pp_exact":
        return ParallelPPExactVaEstimator()
    raise ValueError(
        f"Unknown va_method {method!r}. Use vanilla, backward, parallel_pf, parallel_pp, or parallel_pp_exact."
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
    raise ValueError(
        f"Unknown va_method {method!r}. Use vanilla, parallel_pf, or parallel_pp."
    )
