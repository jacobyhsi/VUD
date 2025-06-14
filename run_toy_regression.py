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
from src.utils import ToyRegressionUtils, GaussianDistribution, extract
from src.prompt import ToyRegressionPrompt
from src.chat import chat_response_only

pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser(description='Running Toy Classification')

parser.add_argument("--dataset_name", default="linear_regression_1", type=str)
parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B", type=str)
parser.add_argument("--model_port", default="8000", type=str)
parser.add_argument("--model_ip", default="localhost", type=str)

parser.add_argument("--x_row_method", default="x_range")
parser.add_argument("--num_x_samples", default=1, type=int)
parser.add_argument("--x_features", default=None)
parser.add_argument("--x_range", default=None)
parser.add_argument("--x_sample_seed", default=0, type=int)

parser.add_argument("--numpy_seed", default=0, type=int)
parser.add_argument("--data_split_seed", default=0, type=int)
parser.add_argument("--icl_sample_seed", default=0, type=int)

parser.add_argument("--shots", default=3, type=int)
parser.add_argument("--num_permutations", default="5", type=int)
parser.add_argument("--num_modified_z", default=3, type=int)
parser.add_argument("--num_random_z", default=3, type=int)
parser.add_argument("--perturbation_std", default=1.0, type=float)
parser.add_argument("--num_candidates", default=3, type=int)
parser.add_argument("--decimal_places", default=1, type=int)
parser.add_argument("--num_outlier_pairs_to_remove", default=0, type=int)
parser.add_argument("--std_method", default="default", type=str)
parser.add_argument("--u_sample_method", default="llm", type=str)

parser.add_argument("--run_name", default="test")
parser.add_argument("--save_directory", default="other")
parser.add_argument("--x_save_value", default=0, type=int)
parser.add_argument("--num_api_calls_save_value", default=0, type=int)

parser.add_argument("--verbose_output", default=0, type=int)
args = parser.parse_args()

@dataclass
class ToyRegressionExperimentConfig:
    dataset_name: str
    model_name: str
    model_port: str
    model_ip: str
    numpy_seed: int
    data_split_seed: int
    icl_sample_seed: int
    shots: int
    x_row_method: int
    num_x_samples: int
    x_features: str
    x_range: str
    x_sample_seed: int
    num_modified_z: int
    num_random_z: int
    perturbation_std: float
    num_candidates: int
    num_permutations: int
    decimal_places: int
    num_outlier_pairs_to_remove: int
    u_sample_method: str
    std_method: str
    run_name: int
    save_directory: int
    x_save_value: int
    num_api_calls_save_value: int
    verbose_output: int

class ToyRegressionExperiment:
    def __init__(self, config: ToyRegressionExperimentConfig):
        self.config = config
        
        np.random.seed(self.config.numpy_seed)

        self.prompter = ToyRegressionPrompt()
        
        if self.config.num_random_z > self.config.num_modified_z:
            raise ValueError("Number of initial random z values cannot be greater than number of modified z values.")

        self.data_preprocessing()
        
        self.num_api_calls = self.config.num_api_calls_save_value

    def data_preprocessing(self):
        self.data_path = f'datasets_toy_regression/{self.config.dataset_name}'

        data, test_data, self.label_keys = load_dataset(
            data_path=self.data_path,
            data_type='toy_regression',
            data_split_seed=self.config.data_split_seed,
        )

        self.feature_columns = ToyRegressionUtils.get_feature_columns(data)

        print("Features:", self.feature_columns)

        self.x_row = ToyRegressionUtils.create_x_row(
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

        D_rows = data.sample(n=self.config.shots, random_state=self.config.icl_sample_seed)
        self.D_feature_stds = D_rows[self.feature_columns].std().to_numpy()

        self.D_note_label_df = D_rows[['note', 'label']]

        if not os.path.exists(f"results/{self.config.dataset_name}/{self.config.save_directory}"):
            os.makedirs(f"results/{self.config.dataset_name}/{self.config.save_directory}")
        D_rows.to_csv(f"results/{self.config.dataset_name}/{self.config.save_directory}/D_{self.config.run_name}.csv", index=False)
        
        self.max_D_label = D_rows['label'].max()
        self.min_D_label = D_rows['label'].min()
    
    def get_next_z(self, z_idx: int, x_idx: int):
        if z_idx < self.config.num_random_z:
            for _ in range(100):
                new_value = np.random.normal(self.x_row.iloc[x_idx][self.feature_columns].to_numpy(np.float32), self.config.perturbation_std * self.D_feature_stds, len(self.feature_columns))
                new_value = np.round(new_value, self.config.decimal_places)
                if not any(np.array_equal(new_value, previous_z_value) for previous_z_value in self.previous_z_values):
                    self.previous_z_values.append(new_value)
                    break
            
            if z_idx == 0:
                
                dict_data = {feature_column: new_value[i] for i, feature_column in enumerate(self.feature_columns)}
                self.z_data = pd.DataFrame([dict_data])
                self.z_data["note"] = self.z_data.apply(lambda row: ToyRegressionUtils.parse_features_to_note(row, self.feature_columns), axis=1)
                
            else:
                modified_row = self.z_data.loc[z_idx-1].copy()
                modified_row[self.feature_columns] = new_value
                modified_row["note"] = ToyRegressionUtils.parse_features_to_note(modified_row, self.feature_columns)
                
                self.z_data.loc[z_idx] = modified_row
                                
        if z_idx >= self.config.num_random_z:
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
            
            modified_row['note'] = ToyRegressionUtils.parse_features_to_note(modified_row, self.feature_columns)
            
            self.z_data.loc[z_idx] = modified_row
            
    def calculate_gaussian(
        self,
        query_note: str,
        probability_calculated: str,
        icl_z_note: Optional[str]=None,
        icl_u_label: Optional[str|float]=None,
    ):
        # Samples from the distribution
        distribution_samples = []

        successful_seeds = 0
        attempts = 0
        while successful_seeds < self.config.num_permutations + self.config.num_outlier_pairs_to_remove*2 and attempts < 100:
        
            if self.config.verbose_output:
                print(f"\n{probability_calculated} Seed {successful_seeds + 1}/{self.config.num_permutations}")

            try:
                permutation_seed = self.num_api_calls
                
                prompt = self.prompter.get_general_prompt(
                    D_df=self.D_note_label_df,
                    query_note=query_note,
                    permutation_seed=permutation_seed, # to avoid seed collision
                    icl_z_note=icl_z_note,
                    icl_u_label=icl_u_label,
                )
            
                if self.config.verbose_output:
                    print(f"Prompt for {probability_calculated}:")
                    print(prompt)

                # Get the prediction and probabilities from the model
                response = chat_response_only(prompt, seed=permutation_seed, model=self.config.model_name, port=self.config.model_port, ip=self.config.model_ip)
                
                self.num_api_calls += 1     
                attempts += 1        

                sample = extract(response)
                
                if not isinstance(sample, float|int):
                    print(f"Invalid sample for {probability_calculated}: {sample}")
                    raise ValueError(f"Invalid sample for {probability_calculated}: {sample}")
                
                if self.config.verbose_output:
                    print(f"y_sample: {sample}")
                
                distribution_samples.append(sample)

                successful_seeds += 1
                
            except:
                print(f"Call {self.num_api_calls} failed. Restarting for seed {successful_seeds}")   
                        
        if successful_seeds == 0:
            raise ValueError(f"All seeds failed for {probability_calculated}.")      
        
        gaussian = ToyRegressionUtils.gaussian_from_samples(distribution_samples, self.config.num_outlier_pairs_to_remove, self.config.std_method)
        
        if self.config.verbose_output:
            print(f"\nGaussian Approximation for {probability_calculated}: mean = {gaussian.mean}, std = {gaussian.std}")
            
        return gaussian, distribution_samples

    def sample_u_values_uniform(self):
        # Sample u from uniform distribution
        u_samples = []
        successful_seeds = 0
        attempts = 0
        while successful_seeds < self.config.num_permutations and attempts < 100:
            u_sample = np.random.uniform(self.min_D_label, self.max_D_label)
            u_sample = np.round(u_sample, self.config.decimal_places)
            if u_sample not in u_samples:
                u_samples.append(u_sample)
                successful_seeds += 1
            attempts += 1
        
        if successful_seeds == 0:
            raise ValueError(f"All seeds failed for u samples.")
        
        if self.config.verbose_output:
            print(f"u_samples: {u_samples}")
        
        return u_samples 
            
    def process_single_x_value(self, x_idx: int):
        self.previous_z_values = []

        self.z_BO_maximisation_objective = []
    
        x = self.x_row['note'].iloc[x_idx]
        x_y = self.x_row['label'].iloc[x_idx]
        print("x:", x)
        
        # Compute p(y|x,D)
        pyx_gaussian, _ = self.calculate_gaussian(x, "p(y|x,D)")
        Hyx = np.round(pyx_gaussian.entropy,5)
        total_variance = np.round(pyx_gaussian.std**2, 5)
                
        save_dict_list = []
            
        for i in tqdm(range(self.config.num_modified_z)):

            self.get_next_z(i, x_idx)
            
            row = self.z_data.iloc[i]
            
            z = row['note']
            
            # Compute p(u|z,D)
            if self.config.u_sample_method == "llm":
                _, u_samples = self.calculate_gaussian(z, "p(u|z,D)", icl_z_note=z)
            elif self.config.u_sample_method == "uniform":
                u_samples = self.sample_u_values_uniform()
            else:
                raise ValueError(f"Invalid u sample method: {self.config.u_sample_method}")
                        
            # Compute p(y|x,u,z,D)
            pyxuz_distributions: list[GaussianDistribution] = []
            Hyxuz = []
            stds = []
            variances = []
            for u_sample in u_samples:
                pyxuz_gaussian, _ = self.calculate_gaussian(x, "p(y|x,u,z,D)", icl_z_note=z, icl_u_label=u_sample)
                Hyxuz.append(pyxuz_gaussian.entropy)
                stds.append(pyxuz_gaussian.std)
                variances.append(pyxuz_gaussian.std**2)
                pyxuz_distributions.append(pyxuz_gaussian)
                
            # Approximate p(y|x,z,D) samples
            pyxuz_samples = []
            for _ in range(100):
                u_sample = np.random.randint(len(u_samples))   
                pyxuz_sample = pyxuz_distributions[u_sample].sample()    
                pyxuz_samples.append(pyxuz_sample)        
            pyxz_gaussian = ToyRegressionUtils.gaussian_from_samples(pyxuz_samples)
                            
            # Entropy
            Hyxz = np.mean(Hyxuz)
            yxz_variance = np.mean(variances)
            yxz_std = np.mean(stds)
            Va = np.round(Hyxz, 5)
            Ve = Hyx - Va
            Va_variance = np.round(yxz_variance, 5)
            Ve_variance = np.round(total_variance - Va_variance, 5)
            
            # KL Divergence
            kl_pyx_pyxz = ToyRegressionUtils.calculate_kl_divergence(pyx_gaussian, pyxz_gaussian)
            kl_pyxz_pyx = ToyRegressionUtils.calculate_kl_divergence(pyxz_gaussian, pyx_gaussian)
            
            self.z_BO_maximisation_objective.append(-Va - kl_pyx_pyxz)
        
            # Save            
            save_dict = {f"z_{feature}": row[feature] for feature in self.feature_columns}
            save_dict["z_note"] = z
            save_dict_x = {f"x_{feature}": self.x_row.iloc[x_idx][feature] for feature in self.feature_columns}
            save_dict_x["x_note"] = x
            save_dict = {**save_dict, **save_dict_x}
            
            save_dict[f"p(y|x,D)_mean"] = pyx_gaussian.mean
            save_dict[f"p(y|x,D)_std"] = pyx_gaussian.std
            save_dict[f"p(y|x,z,D)_mean"] = pyxz_gaussian.mean
            save_dict[f"p(y|x,z,D)_std"] = pyxz_gaussian.std
            save_dict["H[p(y|x,D)]"] = Hyx
            save_dict["Var[y|x,D]"] = total_variance
            save_dict["Va"] = Va
            save_dict["Ve"] = Ve
            save_dict["Va_variance"] = Va_variance
            save_dict["Ve_variance"] = Ve_variance
            save_dict["yxz_std"] = yxz_std
            save_dict["kl_pyx_pyxz"] = kl_pyx_pyxz
            save_dict["kl_pyxz_pyx"] = kl_pyxz_pyx
            save_dict["api_calls"] = self.num_api_calls
            
            save_dict_list.append(save_dict)
            
        save_df = pd.DataFrame(save_dict_list)
        
        return save_df
            
    def run_experiment(self):
        for x_idx in range(self.num_x_values):
            save_df = self.process_single_x_value(x_idx)
            save_df.to_csv(f"results/{self.config.dataset_name}/{self.config.save_directory}/results_{self.config.run_name}_x{x_idx + self.config.x_save_value}.csv", index=False)
        
def main():
    config = ToyRegressionExperimentConfig(**vars(args))
    
    experiment = ToyRegressionExperiment(config)
    
    experiment.run_experiment()

if __name__ == "__main__":
    main()