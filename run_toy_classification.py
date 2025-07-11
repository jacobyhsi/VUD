import os
import re
import argparse
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass

from src.dataset import load_dataset
from src.bayesian_optimisation import new_candidate
from src.utils import ToyClassificationUtils, calculate_entropy, calculate_kl_divergence, calculate_discrete_variance
from src.prompt import ToyClassificationPrompt
from src.chat import chat

pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser(description='Running Toy Classification')

"""LLM API Configuration"""
parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B", type=str)
parser.add_argument("--model_port", default="8000", type=str)
parser.add_argument("--model_ip", default="localhost", type=str)
parser.add_argument("--model_temperature", default=1, type=float)
parser.add_argument("--is_local_client", default=1, type=int)

"""Dataset Configuration"""
parser.add_argument("--dataset_name", default="logistic_regression")
parser.add_argument("--D_size", default=15, type=int)

"""X Configuration"""
parser.add_argument("--x_row_method",type=str, default="x_range")
parser.add_argument("--num_x_samples", default=1, type=int)
parser.add_argument("--x_features", default=None)
parser.add_argument("--x_range", default="{'x1': [-12, 12, 0.2]}")
parser.add_argument("--x_sample_seed", default=0, type=int)
parser.add_argument("--decimal_places", default=1, type=int)

"""Seed Configuration"""
parser.add_argument("--numpy_seed", default=0, type=int)
parser.add_argument("--data_split_seed", default=0, type=int)
parser.add_argument("--icl_sample_seed", default=0, type=int)
parser.add_argument("--fixed_permutation_seed", default=0, type=int)

"""Permutation Related Configuration"""
parser.add_argument("--num_permutations", default=10, type=int)
parser.add_argument("--permute_context", default=1, type=int)

"""Z Configuration"""
parser.add_argument("--num_z", default=15, type=int)
parser.add_argument("--perturb_about_x", default=1, type=int)
parser.add_argument("--perturbation_std", default=0.1, type=float)
parser.add_argument("--num_bo_z", default=0, type=int)
parser.add_argument("--num_candidates", default=3, type=int)

"""Save Configuration"""
parser.add_argument("--run_name", default="test")
parser.add_argument("--save_directory", default="other")
parser.add_argument("--x_save_value", default=0, type=int)
parser.add_argument("--num_api_calls_save_value", default=0, type=int)

parser.add_argument("--verbose_output", default=0, type=int)
args = parser.parse_args()

@dataclass
class ToyClassificationExperimentConfig:
    model_name: str
    model_port: str
    model_ip: str
    model_temperature: float
    is_local_client: int
    
    dataset_name: str
    D_size: int

    x_row_method: str
    num_x_samples: int
    x_features: str
    x_range: str
    x_sample_seed: int
    decimal_places: int
    
    numpy_seed: int
    data_split_seed: int
    icl_sample_seed: int
    fixed_permutation_seed: int
    
    num_permutations: int
    permute_context: int
    
    num_z: int
    perturb_about_x: int
    perturbation_std: float
    num_bo_z: int
    num_candidates: int

    run_name: int
    save_directory: int
    x_save_value: int
    num_api_calls_save_value: int
    
    verbose_output: int
    
class ToyClassificationExperiment:
    def __init__(self, config: ToyClassificationExperimentConfig):
        self.config = config
        
        np.random.seed(self.config.numpy_seed)

        self.prompter = ToyClassificationPrompt()
        
        if self.config.num_bo_z > self.config.num_z:
            raise ValueError("Number of bo z values cannot be greater than number of z values.")
        if self.config.num_bo_z < 0:
            raise ValueError("Number of bo z values cannot be negative.")

        self.data_preprocessing()
        
        self.num_api_calls = self.config.num_api_calls_save_value

    def data_preprocessing(self):
        self.data_path = f'datasets_toy_classification/{self.config.dataset_name}'

        data, test_data, self.label_keys = load_dataset(
            data_path=self.data_path,
            data_type='toy_classification',
            data_split_seed=self.config.data_split_seed,
        )

        self.feature_columns = ToyClassificationUtils.get_feature_columns(data)

        print("Features:", self.feature_columns)

        self.x_row = ToyClassificationUtils.create_x_row(
            method_name=self.config.x_row_method,
            x_features=self.config.x_features,
            x_range=self.config.x_range,
            feature_columns=self.feature_columns,
            decimal_places=self.config.decimal_places,
            num_x_samples=self.config.num_x_samples,
            test_data=test_data,
            x_sample_seed=self.config.x_sample_seed,
        )
            
        self.num_x_values = len(self.x_row)

        D_rows = data.sample(n=self.config.D_size, random_state=self.config.icl_sample_seed)
        self.D_feature_means = D_rows[self.feature_columns].mean().to_numpy()
        self.D_feature_stds = D_rows[self.feature_columns].std().to_numpy()

        self.D_note_label_df = D_rows[['note', 'label']]

        if not os.path.exists(f"results/{self.config.dataset_name}/{self.config.save_directory}"):
            os.makedirs(f"results/{self.config.dataset_name}/{self.config.save_directory}")
        D_rows.to_csv(f"results/{self.config.dataset_name}/{self.config.save_directory}/D_{self.config.run_name}.csv", index=False)
    
    def calculate_avg_probs(
        self,
        query_note: str,
        probability_calculated: str,
        icl_z_note: Optional[str]=None,
        icl_u_label: Optional[str|int]=None,
    ):
        # Initialize p(y|x)
        avg_probs = {label: 0.0 for label in self.label_keys}
        # ----- Processing p(y|x) -----
        successful_seeds = 0
        for seed in range(self.config.num_permutations):
        
            # p(y|x)
            if self.config.verbose_output:
                print(f"\n{probability_calculated} Seed {seed + 1}/{self.config.num_permutations}")
            
            try:
                permutation_seed = self.num_api_calls        
        
                prompt = self.prompter.get_general_prompt(
                    D_df=self.D_note_label_df,
                    query_note=query_note,
                    permutation_seed=permutation_seed if self.config.permute_context else self.config.fixed_permutation_seed,
                    icl_z_note=icl_z_note,
                    icl_u_label=icl_u_label,
                )
                
                if self.config.verbose_output:
                    print(f"Prompt for {probability_calculated}:")
                    print(prompt)

                # Get the prediction and probabilities from the model
                pred, probs = chat(prompt, self.label_keys, seed=permutation_seed, model=self.config.model_name, port=self.config.model_port, ip=self.config.model_ip, temperature=self.config.model_temperature, is_local_client=self.config.is_local_client)
                                
                # Accumulate probabilities
                for label, prob in probs.items():
                    avg_probs[label] += prob
                    
                successful_seeds += 1
            except:
                print(f"Seed {seed + 1} failed.")

            self.num_api_calls += 1

        avg_probs = {label: prob / successful_seeds for label, prob in avg_probs.items()}
        
        if self.config.verbose_output:
            print(f"\nAveraged {probability_calculated} probabilities: {avg_probs}")
            
        return avg_probs
    
    def get_next_z(self, z_idx: int, x_idx: int):
        if z_idx < self.config.num_z - self.config.num_bo_z:
            new_value = np.zeros(len(self.feature_columns), dtype=np.float32)
            for _ in range(100):
                if self.config.perturb_about_x:
                    new_value = np.random.normal(
                        self.x_row.iloc[x_idx][self.feature_columns].to_numpy(np.float32),
                        self.config.perturbation_std * self.D_feature_stds,
                        len(self.feature_columns)
                    )
                else:
                    new_value = np.random.normal(
                        self.D_feature_means,
                        self.config.perturbation_std * self.D_feature_stds,
                        len(self.feature_columns)
                    )
                new_value = np.round(new_value, self.config.decimal_places)
                if not any(np.array_equal(new_value, previous_z_value) for previous_z_value in self.previous_z_values):
                    self.previous_z_values.append(new_value)
                    break
            
            if z_idx == 0:
                
                dict_data = {feature_column: new_value[i] for i, feature_column in enumerate(self.feature_columns)}
                self.z_data = pd.DataFrame([dict_data])
                self.z_data["note"] = self.z_data.apply(lambda row: ToyClassificationUtils.parse_features_to_note(row, self.feature_columns), axis=1)
                
            else:
                modified_row = self.z_data.loc[z_idx-1].copy()
                modified_row[self.feature_columns] = new_value
                modified_row["note"] = ToyClassificationUtils.parse_features_to_note(modified_row, self.feature_columns)
                
                self.z_data.loc[z_idx] = modified_row
                                
        if z_idx >= self.config.num_z - self.config.num_bo_z:
            # Bayesian Optimization for new z values
    
            new_values = new_candidate(
                z_values=self.previous_z_values,
                maximisation_quantity=self.z_BO_maximisation_objective,
                lower_bound=self.x_row.iloc[x_idx][self.feature_columns].to_numpy(np.float32) - 2*self.D_feature_stds,
                upper_bound=self.x_row.iloc[x_idx][self.feature_columns].to_numpy(np.float32) + 2*self.D_feature_stds,
                num_candidates=self.config.num_candidates,
            )
            
            new_values = np.round(new_values, self.config.decimal_places)
            
            new_value = None
            
            for test_value in new_values:
                if not any(np.array_equal(test_value, previous_z_value) for previous_z_value in self.previous_z_values):
                    new_value = test_value
                    break
                else:
                    if self.config.verbose_output:
                        print(f"Duplicate Candidate: {test_value}")            
            if new_value is None:
                if self.config.verbose_output:
                    print("No new candidate found. Using first candidate.")
                new_value = new_values[0]
            
            if self.config.verbose_output:
                print(f"New Z Value: {new_value}")
            self.previous_z_values.append(new_value)
            
            modified_row = self.z_data.loc[z_idx-1].copy()
            modified_row[self.feature_columns] = new_value
            
            modified_row['note'] = ToyClassificationUtils.parse_features_to_note(modified_row, self.feature_columns)
            
            self.z_data.loc[z_idx] = modified_row
            
    def process_single_x_value(self, x_idx: int):
        self.previous_z_values = []

        self.z_BO_maximisation_objective = []
    
        x = self.x_row['note'].iloc[x_idx]
        x_y = self.x_row['label'].iloc[x_idx]
        print("x:", x)
        
        # Compute p(y|x,D)
        avg_pyx_probs = self.calculate_avg_probs(x, "p(y|x,D)")
        Hyx = calculate_entropy(avg_pyx_probs)
        total_variance = calculate_discrete_variance(avg_pyx_probs)
                
        save_dict_list = []
            
        for i in tqdm(range(self.config.num_z)):

            self.get_next_z(i, x_idx)
            
            row = self.z_data.iloc[i]
            
            z = row['note']
            
            # Compute p(u|z,D)
            avg_puz_probs = self.calculate_avg_probs(z, "p(u|z,D)")
            
            # Compute p(y|x,u,z,D)
            avg_pyxu_z_probs = {}
            
            for outer_label in self.label_keys:
                probability_calculated = f"p(y|x,u={outer_label},z,D)"
                
                avg_probs_for_outer_label = self.calculate_avg_probs(
                    query_note=x,
                    probability_calculated=probability_calculated,
                    icl_z_note=z,
                    icl_u_label=outer_label
                )
                
                avg_pyxu_z_probs.update({probability_calculated: avg_probs_for_outer_label})
            
            # Marginalisation
            avg_pyxz_probs = {}

            for label in self.label_keys:  # Iterate over all possible values of y
                avg_pyxz_probs[label] = sum(
                    avg_pyxu_z_probs[f"p(y|x,u={u_label},z,D)"][label] * avg_puz_probs[u_label]
                    for u_label in self.label_keys
                )
                
            # Entropy
            Huz = calculate_entropy(avg_puz_probs)
            Var_uz = calculate_discrete_variance(avg_puz_probs)
            Hyxuz = {f"H[{key}]": calculate_entropy(value) for key, value in avg_pyxu_z_probs.items()}
            Var_yxuz = {f"Var[{key}]": calculate_discrete_variance(value) for key, value in avg_pyxu_z_probs.items()}          
            E_Hyxz = 0.0
            E_Var_yxuz = 0.0
            for label in self.label_keys:
                E_Hyxz += Hyxuz[f"H[p(y|x,u={label},z,D)]"]*avg_puz_probs[label]
                E_Var_yxuz += Var_yxuz[f"Var[p(y|x,u={label},z,D)]"]*avg_puz_probs[label]
            Va = np.round(E_Hyxz, 5)
            Ve = Hyx - Va
            Va_variance = np.round(E_Var_yxuz, 5)
            Ve_variance = total_variance - Va_variance
            
            # KL Divergence
            kl_pyx_pyxz = calculate_kl_divergence(avg_pyx_probs, avg_pyxz_probs)
            kl_pyxz_pyx = calculate_kl_divergence(avg_pyxz_probs, avg_pyx_probs)
            
            self.z_BO_maximisation_objective.append(-Va - kl_pyx_pyxz)
        
            # Save            
            save_dict = {f"z_{feature}": row[feature] for feature in self.feature_columns}
            save_dict["z_note"] = z
            save_dict_x = {f"x_{feature}": self.x_row.iloc[x_idx][feature] for feature in self.feature_columns}
            save_dict_x["x_note"] = x
            save_dict = {**save_dict, **save_dict_x}
            for label, prob in avg_pyx_probs.items():
                save_dict[f"p(y={label}|x,D)"] = prob
            for label, prob in avg_puz_probs.items():
                save_dict[f"p(u={label}|z,D)"] = prob
            for key, outer_label_probs in avg_pyxu_z_probs.items():
                for label, prob in outer_label_probs.items():
                    new_key = re.sub(r'y', f'y={label}', key, count=1)
                    save_dict[new_key] = prob
            for label, prob in avg_pyxz_probs.items():
                save_dict[f"p(y={label}|x,z,D)"] = prob
            save_dict["H[p(u|z,D)]"] = Huz
            save_dict["Var[u|z,D]"] = Var_uz
            for key, entropy in Hyxuz.items():
                save_dict[key] = entropy
            for key, variance in Var_yxuz.items():
                save_dict[key] = variance
            save_dict["H[p(y|x,D)]"] = Hyx
            save_dict["Var[y|x,D]"] = total_variance
            save_dict["Va"] = Va
            save_dict["Ve"] = Ve
            save_dict["Va_variance"] = Va_variance
            save_dict["Ve_variance"] = Ve_variance
            save_dict["kl_pyx_pyxz"] = kl_pyx_pyxz
            save_dict["kl_pyxz_pyx"] = kl_pyxz_pyx
            save_dict["api_calls"] = self.num_api_calls
            
            save_dict_list.append(save_dict)
            
        save_df = pd.DataFrame(save_dict_list)
        
        return save_df
            
    def run_experiment_default(self):
        for x_idx in range(self.num_x_values):
            save_df = self.process_single_x_value(x_idx)
            save_df.to_csv(f"results/{self.config.dataset_name}/{self.config.save_directory}/results_{self.config.run_name}_x{x_idx + self.config.x_save_value}.csv", index=False)
    
    def run_experiment(self):
        self.run_experiment_default()
        
        print(f"Total API Calls: {self.num_api_calls}")
        
        with open(f"results/{self.config.dataset_name}/{self.config.save_directory}/api_calls_{self.config.run_name}.txt", "w") as f:
            f.write(f"Total API Calls: {self.num_api_calls}")
def main():
    config = ToyClassificationExperimentConfig(**vars(args))
    
    experiment = ToyClassificationExperiment(config)
    
    experiment.run_experiment()

if __name__ == "__main__":
    main()