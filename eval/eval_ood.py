import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Utility function to compute metrics and lengths
def compute_metrics(filepaths):
    aurocs_total = []
    aurocs_epi = []
    accs = []
    lengths = []
    for path in filepaths:
        df = pd.read_csv(path)
        df = pd.read_csv(path).head(120)
        ve_id = df[df["is_ood"] == 0]["Ve"].mean()
        ve_ood = df[df["is_ood"] == 1]["Ve"].mean()
        print(f"  Mean Ve (ID):                  {ve_id:.3f}")
        print(f"  Mean Ve (OOD):                 {ve_ood:.3f}")
        print(f"  Change: {ve_id - ve_ood:.3f}")
        labels = df["is_ood"].values
        aurocs_total.append(roc_auc_score(labels, df["TU"].values))
        aurocs_epi.append(roc_auc_score(labels, df["Ve"].values))
        accs.append(accuracy_score(df["true_label"], df["pred_label"]))
        lengths.append(len(df))
    return (
        np.array(aurocs_total),
        np.array(aurocs_epi),
        np.array(accs),
        lengths
    )

file_groups = {
    ("BoolQA", "HotpotQA"): [
        "results_qa/[FILE_NAME_IN_results_qa].csv",
    ],
    ("BoolQA", "PubMedQA"): [
        "results_qa/[FILE_NAME_IN_results_qa].csv",
    ],
    ("HotpotQA", "BoolQA"): [
        "results_qa/[FILE_NAME_IN_results_qa].csv",
    ],
    ("HotpotQA", "PubMedQA"): [
        "results_qa/[FILE_NAME_IN_results_qa].csv",
    ],
    ("PubMedQA", "BoolQA"): [
        "results_qa/[FILE_NAME_IN_results_qa].csv",
    ],
    ("PubMedQA", "HotpotQA"): [
        "results_qa/[FILE_NAME_IN_results_qa].csv",
    ],
}

# Compute and print results
for (id_name, ood_name), paths in file_groups.items():
    try:
        print(f"{id_name} (ID) → {ood_name} (OOD)")
        aurocs_total, aurocs_epi, accs, lengths = compute_metrics(paths)
        print(f"  Accuracy:                     {accs.mean():.3f} ± {accs.std():.3f}")
        print(f"  AUROC (Total Uncertainty):    {aurocs_total.mean():.3f} ± {aurocs_total.std():.3f}")
        print(f"  AUROC (Epistemic Uncertainty):{aurocs_epi.mean():.3f} ± {aurocs_epi.std():.3f}")
        for path, length in zip(paths, lengths):
            print(f"{length} samples")
        print()
    except FileNotFoundError as e:
        print(f"{id_name} (ID) → {ood_name} (OOD): File not found -> {e.filename}\n")
