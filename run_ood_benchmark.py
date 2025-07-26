import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.dataset import QADataset
from src.prompt import Prompt
from src.chat import chat_qa
from src.utils import calculate_entropy

# OOD Benchmark for Question Answering (QA) tasks using Deep Ensembles

def sample_in_context_set(df_train: pd.DataFrame, num_d: int, seed: int) -> str:
    """Return a single string containing `num_d` randomly‑sampled in‑context rows.

    The returned string is suitable for Prompt.get_pyxD_prompt().
    """
    df_D = df_train.sample(n=num_d, random_state=seed).reset_index(drop=True)
    return "\n".join(
        f"{row['note']} <output>{row['label']}</output>\n" for _, row in df_D.iterrows()
    )


def aggregate_distributions(dist_list):
    """Average a list of dictionaries with identical keys → numpy array + key order."""
    keys = list(dist_list[0].keys())
    arr = np.array([[d[k] for k in keys] for d in dist_list])  # shape (K, |Y|)
    return arr.mean(axis=0), keys, arr


def main():
    global_seed = int(args.seed)

    qa = QADataset(args.id, args.ood)
    train_id, test_id, test_ood, label_keys = qa.load_data()

    # flag OOD vs ID
    test_id["is_ood"] = 0
    test_ood["is_ood"] = 1

    interleaved_rows = [row for pair in zip(test_id.iterrows(), test_ood.iterrows()) for row in pair]
    test_df = pd.DataFrame([row[1] for row in interleaved_rows]).head(120)

    # test_df = pd.concat([test_id, test_ood], ignore_index=True).head(120)
    # test_df = test_df.sample(n=25, random_state=global_seed).reset_index(drop=True)

    prompt_builder = Prompt(prompt_type="tabular")

    results = []
    for idx, x_row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test examples"):
        x_note = x_row["note"]
        true_label = x_row["label"]
        is_ood = x_row["is_ood"]

        # ------------------------------------------------------------------
        # Build K prompts with different in‑context sets and query LLM
        # ------------------------------------------------------------------
        distributions = []  # list of dicts p_k(y|x, D_k)
        for k in range(args.K):
            seed_k = global_seed + k  # sampling D across different seeds whereas our method only samples D once with a fixed seed
            D_str = sample_in_context_set(train_id, args.num_d, seed=seed_k)
            prompt_pyxD = prompt_builder.get_pyxD_prompt(x_note, D_str)

            # Query
            output, p_yxD = chat_tabular(prompt_pyxD, label_keys, seed_k)
            # Guard‑rails: if parsing fails, resample once (optional)
            if not p_yxD:
                continue  # skip this ensemble member
            distributions.append(p_yxD)

        if len(distributions) == 0:
            # Could not obtain any valid prediction; skip example
            continue

        # If <K valid members, we still proceed with whatever we have
        q_ens, key_order, arr_K = aggregate_distributions(distributions)

        # ------------------------------------------------------------------
        # Uncertainty decomposition
        # ------------------------------------------------------------------
        U_total = calculate_entropy({k: v for k, v in zip(key_order, q_ens)})
        U_alea = np.mean([calculate_entropy(d) for d in distributions])
        U_epi = U_total - U_alea

        # Predicted label = argmax ensemble prob
        pred_label = key_order[int(np.argmax(q_ens))]

        # Store
        results.append({
            "x_note": x_note,
            "TU": U_total,
            "Va": U_alea,  # using Va to keep column naming from original script
            "Ve": U_epi,
            "true_label": true_label,
            "pred_label": pred_label,
            "is_ood": is_ood
        })

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    out_name = (
        f"results_qa/ensembles2/df_{args.id}_ID_{args.ood}_OOD_"
        f"{len(df_results)}x_{args.num_d}D_{args.K}Ens_seed{global_seed}_seed{args.run_seed}.csv"
    )
    df_results.to_csv(out_name, index=False)
    print("Saved →", out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Deep Ensembles* QA OOD Pipeline")
    parser.add_argument("--seed", type=int, default=123, help="Global random seed")
    parser.add_argument("--id", default="boolqa", help="In domain dataset name")
    parser.add_argument("--ood", default="pubmedqa", help="OOD dataset name")
    parser.add_argument("--num_d", type=int, default=15, help="# in context examples per prompt")
    parser.add_argument("--K", type=int, default=5, help="# ensemble members (prompts) per x")
    parser.add_argument("--max_examples", type=int, default=None, help="Debug limit on #test rows")
    parser.add_argument("--run_seed", type=int, default=0, help="Random seed for sampling test rows")
    args = parser.parse_args()
    main()
