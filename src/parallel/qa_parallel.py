"""QA Parallel PF and discrete Parallel PP estimators (bits).

The joint sampler is the same constrained entropy-order two-slot denoiser as
the toy parallel method, with the full QA prompt and Instruct chat template.
PF scores with the forward prompt; PP estimates H_P(Y|U) from joint counts.
Both retain the separately prompted TU baseline, so Ve need not be positive.
"""

import json
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.chat import chat_qa, sample_joint_labels
from src.dream_backend import U_PLACEHOLDER, is_dream_model
from src.utils import calculate_entropy, calculate_kl_divergence
from src.parallel.parallel_stats import (
    PP_ESTIMATOR,
    canonical_parallel_method,
    joint_count_statistics,
)


ESTIMATOR = "qa_mc_joint_sample_conditional_nll_v1"


def parallel_estimator_id(method):
    return PP_ESTIMATOR if canonical_parallel_method(method) == "parallel_pp" else ESTIMATOR


def validate_parallel_checkpoint(existing, args):
    method = canonical_parallel_method(args.va_method)
    allowed_methods = {method, "parallel"} if method == "parallel_pf" else {method}
    if "va_method" not in existing or not existing["va_method"].isin(allowed_methods).all():
        raise ValueError("Parallel checkpoint configuration mismatch: va_method.")
    expected = {
        "parallel_joint_estimator": parallel_estimator_id(method),
        "n_joint_samples": args.num_joint_samples,
        "num_seeds": args.num_seeds, "seed": args.seed,
    }
    for key, value in expected.items():
        if key not in existing or not existing[key].eq(value).all():
            raise ValueError(f"Parallel checkpoint configuration mismatch: {key}.")


def validate_parallel_args(args):
    if canonical_parallel_method(args.va_method) not in {"parallel_pf", "parallel_pp"}:
        raise ValueError("Expected parallel_pf or parallel_pp (legacy parallel means PF).")
    if not is_dream_model(args.model):
        raise ValueError("QA --va_method parallel requires a Dream model.")
    if args.num_joint_samples < 1 or args.num_seeds < 1:
        raise ValueError("--num_joint_samples and --num_seeds must be positive.")


def parallel_output_dir(args):
    # Preserve paths for already queued legacy commands; explicit names use
    # new, distinct paths. Existing result files are never migrated or renamed.
    directory = "va_parallel_entropic_mc" if args.va_method == "parallel" else f"va_{args.va_method}"
    return (
        f"results/qa/{args.model.rsplit('/', 1)[-1]}/{directory}/"
        f"n{args.num_joint_samples}_seed{args.seed}_permutations{args.num_seeds}"
    )


def _icl(df, seed):
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Keep the existing QA runner's whitespace and example ordering convention.
    return "\n".join(
        f"{row['note']} <output>{row['label']}</output>\n"
        for _, row in shuffled.iterrows()
    )


def _score(message, labels, seed, args):
    _, probs = chat_qa(
        message, labels, seed, model=args.model, port=args.port, ip=args.host,
    )
    probs = {str(k): float(v) for k, v in probs.items()}
    if set(probs) != set(labels) or any(
        not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()
    ) or not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5):
        raise ValueError(f"Invalid QA label probabilities: {probs}")
    return probs


def estimate_parallel_z(x, z, df_D, prompt, labels, baseline, args, *, x_index, z_index):
    """N total draws per z, cycling through num_seeds ICL permutations.

    In PF each sampled pair is scored with its own context permutation, without
    averaging probabilities before the log. Deterministic forward scores are
    cached by (permutation, u). Errors fail the run rather than rejecting and
    resampling difficult pairs, which would change the joint sampling law.
    PP pools joint counts across permutations and makes no conditional calls.
    """
    validate_parallel_args(args)
    is_pp = canonical_parallel_method(args.va_method) == "parallel_pp"
    labels = [str(k) for k in labels]
    augmented = pd.concat([
        pd.DataFrame([{"note": z, "label": U_PLACEHOLDER}]), df_D,
    ], ignore_index=True)
    messages = {}
    scores = {}
    counts = {u: {y: 0 for y in labels} for u in labels}
    nlls = []
    for sample_index in range(args.num_joint_samples):
        context_seed = sample_index % args.num_seeds
        if context_seed not in messages:
            icl = _icl(augmented, context_seed)
            messages[context_seed] = (icl, prompt.get_pyxuzD_prompt(x, icl))
        icl, message = messages[context_seed]
        if message.count(U_PLACEHOLDER) != 1:
            raise ValueError("QA joint prompt must contain exactly one auxiliary placeholder.")
        # Stable across process restarts/resume and independent across x/z/draw.
        joint_seed = int(np.random.SeedSequence([
            args.seed, int(x_index), int(z_index), sample_index,
        ]).generate_state(1)[0])
        u, y = sample_joint_labels(
            icl, x, labels, seed=joint_seed, temperature=1.0,
            model=args.model, message=message,
        )
        u, y = str(u), str(y)
        if u not in counts or y not in counts[u]:
            raise ValueError(f"Joint sampler returned invalid QA labels {(u, y)!r}.")
        counts[u][y] += 1
        if is_pp:
            continue
        key = (context_seed, u)
        if key not in scores:
            # Identical QA prompt, sampled u filled in, y scored by vanilla.
            scores[key] = _score(
                message.replace(U_PLACEHOLDER, u), labels, context_seed, args,
            )
        probability = scores[key][y]
        if probability <= 0:
            raise ValueError(
                f"Forward p(y={y}|u={u}) is zero: NLL is infinite. "
                "Refusing to discard/resample the joint pair."
            )
        nlls.append(-math.log2(probability))

    stats = joint_count_statistics(counts)
    joint, marginal_y = stats["joint"], stats["py"]
    result = {
        "z_note": z,
        "Va": stats["Va"] if is_pp else float(np.mean(nlls)),
        "KL": calculate_kl_divergence(marginal_y, baseline),
        "parallel_joint": joint,
        "parallel_y": marginal_y,
    }
    if is_pp:
        result.update(parallel_counts=counts, parallel_u_counts=stats["u_counts"],
                      parallel_conditional=stats["conditional"])
    return result


def estimate_parallel_qa(x_row, data_z, df_D, prompt, label_keys, args, *, x_index):
    validate_parallel_args(args)
    labels = [str(k) for k in label_keys]
    x = x_row["note"]
    # Dream forward scoring is deterministic: compute TU once per x, with the
    # same D permutations 0,...,num_seeds-1 as vanilla's successful seed loop.
    baseline = {k: 0.0 for k in labels}
    for seed in range(args.num_seeds):
        probs = _score(prompt.get_pyxD_prompt(x, _icl(df_D, seed)), labels, seed, args)
        for k in labels:
            baseline[k] += probs[k] / args.num_seeds

    candidates = []
    for z_index, (_, row) in enumerate(tqdm(
        data_z.iterrows(), total=len(data_z), desc="Processing parallel z",
    )):
        candidates.append(estimate_parallel_z(
            x, row["note"], df_D, prompt, labels, baseline, args,
            x_index=x_index, z_index=z_index,
        ))
    if not candidates:
        raise ValueError("Parallel QA requires at least one z candidate.")
    # Preserve vanilla's top-five-KL, then minimum-Va selection procedure.
    shortlist = sorted(candidates, key=lambda result: result["KL"])[:5]
    selected = min(shortlist, key=lambda result: result["Va"])
    total = calculate_entropy(baseline)
    result = {
        "is_ood": x_row["is_ood"],
        "TU": total,
        "Va": selected["Va"],
        "Ve": total - selected["Va"],
        "true_label": x_row["label"],
        "pred_label": max(baseline, key=baseline.get),
        "x_note": x,
        "z_note": selected["z_note"],
        "KL": selected["KL"],
        "va_method": canonical_parallel_method(args.va_method),
        "parallel_joint_estimator": parallel_estimator_id(args.va_method),
        "n_joint_samples": args.num_joint_samples,
        "num_seeds": args.num_seeds,
        "seed": args.seed,
        "parallel_joint": json.dumps(selected["parallel_joint"]),
        "parallel_y": json.dumps(selected["parallel_y"]),
        "forward_y": json.dumps(baseline),
    }
    if canonical_parallel_method(args.va_method) == "parallel_pp":
        for key in ("parallel_counts", "parallel_u_counts", "parallel_conditional"):
            result[key] = json.dumps(selected[key], allow_nan=False)
    return result
