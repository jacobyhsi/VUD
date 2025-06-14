import re
import argparse
import pandas as pd
from tqdm import tqdm
from src.dataset import QADataset
from src.prompt import Prompt
from src.chat import chat_qa
from src.utils import calculate_entropy, calculate_kl_divergence, QAUtils

# Main
def main():
    global_seed = int(args.seed)
    qa = QADataset(args.id, args.ood)
    train_id, test_id, test_ood, label_keys = qa.load_data()

    print(f"Train shape: {train_id.shape}")
    print(f"Test (ID) shape: {test_id.shape}")
    print(f"Test (OOD) shape: {test_ood.shape}")
    print(f"Label keys: {label_keys}")

    # Sample D
    num_D = 15
    df_D = train_id.sample(n=num_D, random_state=global_seed)
    train_id = train_id.drop(df_D.index)

    # Sample z
    num_z = 20
    df_z = train_id.sample(n=1, random_state=global_seed)
    train_id = train_id.drop(df_z.index)
    
    # Sample x
    test_id["is_ood"] = 0
    test_ood["is_ood"] = 1

    # Interleave rows
    interleaved_rows = [row for pair in zip(test_id.iterrows(), test_ood.iterrows()) for row in pair]
    data_x = pd.DataFrame([row[1] for row in interleaved_rows])
    num_x = len(data_x)

    prompt = Prompt(prompt_type="tabular")

    print(f"Processing: df_{args.id}_ID_{args.ood}_OOD_{num_x}x_{num_z}z_{num_D}ICL_new-z-prompt_bug-amend_instruct_dualgpu.csv")
    results = []
    for i, x_row in tqdm(data_x.iterrows(), total=num_x, desc="Processing x"):
        x = x_row['note']

        min_Va_lst = []
        seed = 0

        data_z = QAUtils.perturb_z(data=df_D, x_row=x_row, z_samples=num_z, seed=seed)
        data_z["puzD"] = None
        data_z["pyxuzD"] = None

        for j, row in tqdm(data_z.iterrows(), total=len(data_z), desc="Processing z"):
            seed = 0
            z = row['note']

            # Initialize dictionaries to store average probabilities
            avg_puzD_probs = {label: 0.0 for label in label_keys}
            avg_pyxuzD_probs = {f"p(y|x,u{outer_label},z,D)": {inner_label: 0.0 for inner_label in label_keys} for outer_label in label_keys}
            avg_pyxzD_probs = {label: 0.0 for label in label_keys}
            avg_pyxD_probs = {label: 0.0 for label in label_keys}

            # Extracting prompt probabilities: p(u|z,D), p(y|x,u,z,D), p(y|x,D)
            successful_seeds = 0
            successful_seeds_lst = []
            
            num_seeds = args.num_seeds  # target number of successful seeds
            while successful_seeds < num_seeds:

                # Create temporary dictionaries for this seed
                temp_avg_puzD = {label: 0.0 for label in label_keys}
                temp_avg_pyxuzD = {f"p(y|x,u{outer_label},z,D)": {inner_label: 0.0 for inner_label in label_keys} for outer_label in label_keys}
                temp_avg_pyxD = {label: 0.0 for label in label_keys}

                ## Shuffle D
                df_D_shuffled = df_D.sample(frac=1, random_state=seed).reset_index(drop=True)
                D = "\n".join(
                    [f"{row['note']} <output>{row['label']}</output>\n" for _, row in df_D_shuffled.iterrows()]
                )
                
                # p(u|z,D)
                prompt_puzD = prompt.get_puzD_prompt(z, D)
                output_puzD, puzD = chat_qa(prompt_puzD, label_keys, seed)
                if not re.search(r'\d+</output>', output_puzD):
                    seed += 1
                    continue
                if not puzD:
                    seed += 1
                    continue
                for label, prob in puzD.items():
                    temp_avg_puzD[label] += prob

                # p(y|x,u,z,D)
                skip_seed = False
                dict_uz = {}
                for label_key in label_keys:
                    df_copy = df_z.copy()
                    df_copy["label"] = label_key
                    dict_uz[f"u{label_key}z"] = df_copy

                dict_uzD = {}
                for key, df_uz in dict_uz.items():
                    dict_uzD[f"{key}D"] = pd.concat([df_uz, df_D], ignore_index=True)
                    # Shuffle u,z,D
                    dict_uzD[f"{key}D"] = dict_uzD[f"{key}D"].sample(frac=1, random_state=seed).reset_index(drop=True)

                prompt_uzD = {}
                for key, df_uzD in dict_uzD.items():
                    prompt_uzD[f"{key}"] = "\n".join([f"{row['note']} <output>{row['label']}</output>\n" for _, row in df_uzD.iterrows()])

                for key, icl in prompt_uzD.items():
                    u_value = re.search(r"u(\d+)", key).group(1)  # Match 'u' followed by digits
                    prompt_pyxuzD = prompt.get_pyxuzD_prompt(x, icl)
                    output_pyxuzD, pyxuzD = chat_qa(prompt_pyxuzD, label_keys, seed)
                    if not re.search(r'\d+</output>', output_pyxuzD):
                        skip_seed = True
                        break
                    if not pyxuzD:
                        skip_seed = True
                        break
                    for label, prob in pyxuzD.items():
                        temp_avg_pyxuzD[f"p(y|x,u{u_value},z,D)"][label] += prob
                if skip_seed:
                    seed += 1
                    continue

                # p(y|x,D)
                prompt_pyxD = prompt.get_pyxD_prompt(x, D)
                output_pyxD, pyxD = chat_qa(prompt_pyxD, label_keys, seed)
                if not re.search(r'\d+</output>', output_pyxD):
                    seed += 1
                    continue
                if not pyxD:
                    seed += 1
                    continue
                for label, prob in pyxD.items():
                    temp_avg_pyxD[label] += prob

                # Only update the global accumulators if all outputs are valid
                for label in label_keys:
                    avg_puzD_probs[label] += temp_avg_puzD[label]
                    avg_pyxD_probs[label] += temp_avg_pyxD[label]
                    for u_label in label_keys:
                        avg_pyxuzD_probs[f"p(y|x,u{u_label},z,D)"][label] += temp_avg_pyxuzD[f"p(y|x,u{u_label},z,D)"][label]

                successful_seeds += 1
                successful_seeds_lst.append(seed)
                seed += 1

            # print("Successful Seeds List:", successful_seeds_lst)

            # Average probabilities
            for label in label_keys:
                avg_puzD_probs[label] /= num_seeds
                avg_pyxD_probs[label] /= num_seeds
                for u_label in label_keys:
                    avg_pyxuzD_probs[f"p(y|x,u{u_label},z,D)"][label] /= num_seeds

            # p(y|x,z,D) via marginalization
            for label in label_keys:
                avg_pyxzD_probs[label] = sum(
                    avg_pyxuzD_probs[f"p(y|x,u{u_label},z,D)"][label] * avg_puzD_probs[u_label]
                    for u_label in avg_puzD_probs.keys()
                )

            ## Thresholding p(y|x,z,D) and p(y|x,D) via KL Divergence
            kl = calculate_kl_divergence(avg_pyxzD_probs, avg_pyxD_probs)
            data_z.at[j, 'KL'] = kl
            data_z.at[j, 'puzD'] = puzD.copy()
            data_z.at[j, 'pyxuzD'] = avg_pyxuzD_probs.copy()

        min_kl = data_z.sort_values('KL').head(5)
        for _, row in min_kl.iterrows():
            # Retrieve stored puzD and pyxuzD for this z candidate
            puzD = row['puzD']
            pyxuzD = row['pyxuzD']

            # Optional safety check (strongly recommended)
            assert abs(sum(puzD.values()) - 1.0) < 1e-5
            for key in pyxuzD:
                assert abs(sum(pyxuzD[key].values()) - 1.0) < 1e-5

            # Compute Va for this z
            E_H_pyxuzD = sum(
                calculate_entropy(pyxuzD[f"p(y|x,u{u_label},z,D)"]) * puzD[u_label]
                for u_label in puzD.keys()
            )

            row['Va'] = E_H_pyxuzD
            min_Va_lst.append(row)

        H_pyxD = calculate_entropy(avg_pyxD_probs)

        min_Va = min(min_Va_lst, key=lambda row: row['Va'])
        min_Va['TU'] = H_pyxD

        Ve = H_pyxD - min_Va['Va']
        min_Va['Ve'] = Ve

        pred_label = max(avg_pyxD_probs, key=avg_pyxD_probs.get)
        true_label = x_row['label']
        is_ood = x_row["is_ood"]

        x_z = {
            'is_ood': is_ood,
            'TU': min_Va['TU'],
            'Va': min_Va['Va'],
            'Ve': min_Va['Ve'],
            'true_label': true_label,
            'pred_label': pred_label,
            'x_note': x_row['note']
        }
        x_z = pd.DataFrame([x_z])
        results.append(x_z)

        df_results = pd.concat(results, ignore_index=True)
        df_results.to_csv(f"results_qa/df_{args.id}_ID_{args.ood}_OOD_{num_x}x_{num_z}z_{num_D}ICL.csv", index=False)

if __name__ == "__main__":
    # Argument Parser
    pd.set_option('display.max_columns', None)
    parser = argparse.ArgumentParser(description='Run VPUD')
    parser.add_argument("--seed", default=123)
    parser.add_argument("--id", default="boolqa")
    parser.add_argument("--ood", default="pubmedqa")
    parser.add_argument("--num_seeds", default=5)
    args = parser.parse_args()
    main()
