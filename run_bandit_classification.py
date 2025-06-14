import os
import re
import argparse
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass

from src.bandit import get_bandit, ClassificationBandit, ButtonsBandit
from src.bandit_algorithms import UCB1_Algorithm
from src.bayesian_optimisation import new_candidate
from src.utils import BanditClassificationUtils, calculate_entropy, calculate_kl_divergence, calculate_discrete_mean, calculate_discrete_variance, calculate_min_Va_by_KL_rank
from src.prompt import BanditClassificationPrompt
from src.chat import chat

pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser(description='Running Toy Classification')

parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B", type=str)
parser.add_argument("--model_port", default="8000", type=str)
parser.add_argument("--model_ip", default="localhost", type=str)

parser.add_argument("--bandit_name", default="buttons", type=str)
parser.add_argument("--bandit_num_arms", default=5, type=int)
parser.add_argument("--bandit_midpoint", default=0.5, type=float)
parser.add_argument("--bandit_gap", default=0.2, type=float)
parser.add_argument("--bandit_seed", default=0, type=int)
parser.add_argument("--bandit_exploration_rate", default=2, type=float)

parser.add_argument("--is_contextual_bandit", default=0, type=int)

parser.add_argument("--num_trials", default=10, type=int)
parser.add_argument("--num_random_trials", default=3, type=int)
parser.add_argument("--uncertainty_type", default="epistemic", type=str)

parser.add_argument("--numpy_seed", default=0, type=int)
parser.add_argument("--use_api_call_seed", default=0, type=int)

parser.add_argument("--num_permutations", default=5, type=int)
parser.add_argument("--num_modified_z", default=1, type=int)
parser.add_argument("--num_random_z", default=1, type=int)
parser.add_argument("--perturbation_std", default=1.0, type=float)
parser.add_argument("--num_candidates", default=3, type=int)
parser.add_argument("--decimal_places", default=3, type=int)
parser.add_argument("--min_KL_rank", default=1, type=int)

parser.add_argument("--run_name", default="test")
parser.add_argument("--save_directory", default="other")
parser.add_argument("--num_api_calls_save_value", default=0, type=int)

parser.add_argument("--verbose_output", default=0, type=int)
args = parser.parse_args()

@dataclass
class BanditClassificationExperimentConfig:
    model_name: str
    model_port: str
    model_ip: str
    bandit_name: str
    bandit_num_arms: int
    bandit_midpoint: float
    bandit_gap: float
    bandit_seed: int
    bandit_exploration_rate: float
    is_contextual_bandit: int
    numpy_seed: int
    num_trials: int
    num_random_trials: int
    uncertainty_type: str
    use_api_call_seed: int
    num_modified_z: int
    num_random_z: int
    perturbation_std: float
    num_candidates: int
    num_permutations: int
    decimal_places: int
    min_KL_rank: int
    run_name: int
    save_directory: int
    num_api_calls_save_value: int
    verbose_output: int

class BanditClassificationExperiment:
    def __init__(self, config: BanditClassificationExperimentConfig):
        self.config = config
        
        self.rng = np.random.default_rng(self.config.numpy_seed)

        self.prompter = BanditClassificationPrompt()
        
        if self.config.num_random_z > self.config.num_modified_z:
            raise ValueError("Number of initial random z values cannot be greater than number of modified z values.")

        self.create_bandit()
        
        self.use_api_call_seed = self.config.use_api_call_seed == 1
        self.num_api_calls = self.config.num_api_calls_save_value
        
        if self.config.uncertainty_type == "ucb1":
            self.UCB1_algorithm = UCB1_Algorithm(num_arms=self.config.bandit_num_arms, c=self.config.bandit_exploration_rate)
        else:
            self.UCB1_algorithm = None
            
    def create_bandit(self):
        self.bandit: ClassificationBandit = get_bandit(
            bandit_name=self.config.bandit_name,
            num_arms=self.config.bandit_num_arms,
            gap=self.config.bandit_gap,
            midpoint=self.config.bandit_midpoint,
            seed=self.config.bandit_seed,
        )
        
        self.label_keys = self.bandit.get_reward_space()
        self.action_space = self.bandit.get_action_space()
        
        if isinstance(self.bandit, ButtonsBandit):
            print(f"Best arm: {self.bandit.best_arm}")
            
               
        if self.config.is_contextual_bandit:
            self.feature_columns = self.bandit.get_context_feature_cols()

            print("Features:", self.feature_columns)
        else:
            self.feature_columns = []
            
        self.num_trials = self.config.num_trials
        
        self.D_rows: pd.DataFrame = None
    
    @property
    def D_feature_stds(self):
        self._D_feature_stds = self.D_rows[self.feature_columns].std().to_numpy().flatten()
        return self._D_feature_stds
    
    @property
    def D_note_label_df(self):
        return self.D_rows[['note', 'label']]
    
    def save_D_rows(self):
        if not os.path.exists(f"results/bandits/{self.config.bandit_name}/{self.config.save_directory}"):
            os.makedirs(f"results/bandits/{self.config.bandit_name}/{self.config.save_directory}")
        self.D_rows.to_csv(f"results/bandits/{self.config.bandit_name}/{self.config.save_directory}/D_{self.config.run_name}.csv", index=False)
    
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

            permutation_seed = self.num_api_calls if self.use_api_call_seed else seed
            
            try:
                
                prompt = self.prompter.get_general_prompt(
                    D_df=self.D_note_label_df,
                    query_note=query_note,
                    permutation_seed=permutation_seed,
                    icl_z_note=icl_z_note,
                    icl_u_label=icl_u_label,
                )
                if self.config.verbose_output:
                    print(f"Prompt for {probability_calculated}:")
                    print(prompt)

                # Get the prediction and probabilities from the model
                pred, probs = chat(prompt, self.label_keys, seed=permutation_seed, model=self.config.model_name, port=self.config.model_port, ip=self.config.model_ip)
                
                self.num_api_calls += 1
                
                # Accumulate probabilities
                for label, prob in probs.items():
                    avg_probs[label] += prob
                    
                successful_seeds += 1
            except:
                print(f"Seed {seed + 1} failed.")
        
        avg_probs = {label: prob / successful_seeds for label, prob in avg_probs.items()}
        
        if self.config.verbose_output:
            print(f"\nAveraged {probability_calculated} probabilities: {avg_probs}")
            
        return avg_probs
    
    def get_random_action(self):
        action = self.rng.choice(self.action_space)
        return action
    
    def get_next_z(self, z_idx: int, context: Optional[dict] = None, action: Optional[str|int] = None):
        if action is None: 
            action = self.get_random_action()
            
        for _ in range(100):
            if self.config.is_contextual_bandit:
                new_value = self.rng.normal(np.array([float(x) for x in list(context.values())]),
                    self.config.perturbation_std * self.D_feature_stds, len(self.feature_columns)
                )           
                new_value = np.round(new_value, self.config.decimal_places)
                if not any(np.array_equal(new_value, previous_z_value) for previous_z_value in self.previous_z_values):
                    self.previous_z_values.append(new_value)
                    break
        
        if z_idx == 0:
            dict_data = {}
            if self.config.is_contextual_bandit:
                dict_data = {feature_column: new_value[i] for i, feature_column in enumerate(self.feature_columns)}
            dict_data.update({"action": action})
            self.z_data = pd.DataFrame([dict_data])
            if self.config.is_contextual_bandit:
                self.z_data["note"] = self.z_data.apply(lambda row: BanditClassificationUtils.parse_features_and_action_to_note(row=row, feature_columns=self.feature_columns, action=action, decimal_places=self.config.decimal_places), axis=1)
            else:
                self.z_data["note"] = BanditClassificationUtils.parse_features_and_action_to_note(action=action, decimal_places=self.config.decimal_places)
        else:
            modified_row = self.z_data.loc[z_idx-1].copy()
            modified_row[self.feature_columns] = new_value
            modified_row["action"] = action
            modified_row["note"] = BanditClassificationUtils.parse_features_and_action_to_note(action=action, row=modified_row, feature_columns=self.feature_columns, decimal_places=self.config.decimal_places)
            
            self.z_data.loc[z_idx] = modified_row
        
        
            
    def process_single_trial_action(self, trial: int, action: str|int, context: Optional[pd.Series] = None):
        self.previous_z_values = []

        x = BanditClassificationUtils.parse_features_and_action_to_note(action=action, row=context, feature_columns=self.feature_columns, decimal_places=self.config.decimal_places)
        
        # Compute p(y|x,D)
        avg_pyx_probs = self.calculate_avg_probs(x, "p(y|x,D)")
        total_variance = calculate_discrete_variance(avg_pyx_probs)
        mean_y = calculate_discrete_mean(avg_pyx_probs)
                
        save_dict_list = []
            
        for i in range(self.config.num_modified_z):

            self.get_next_z(z_idx=i, context=context, action=action)
            
            row = self.z_data.iloc[i]
            
            z = row['note']
            
            if self.config.verbose_output:
                print(f"z: {z}")
            
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
                
            # Variance
            Var_uz = calculate_discrete_variance(avg_puz_probs)
            Var_yxuz = {f"Var[{key}]": calculate_discrete_variance(value) for key, value in avg_pyxu_z_probs.items()}          
            E_Var_yxuz = 0.0
            for label in self.label_keys:
                E_Var_yxuz += Var_yxuz[f"Var[p(y|x,u={label},z,D)]"]*avg_puz_probs[label]
            Va_variance = np.round(E_Var_yxuz, 5)
            Ve_variance = total_variance - Va_variance
            
            # KL Divergence
            kl_pyx_pyxz = calculate_kl_divergence(avg_pyx_probs, avg_pyxz_probs)
            kl_pyxz_pyx = calculate_kl_divergence(avg_pyxz_probs, avg_pyx_probs)
                    
            # Save            
            save_dict = {f"z_{feature}": row[feature] for feature in self.feature_columns}
            save_dict["z_note"] = z
            save_dict_x = {f"x_{feature}": context[feature] for feature in self.feature_columns}
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
            save_dict["Var[u|z,D]"] = Var_uz
            for key, variance in Var_yxuz.items():
                save_dict[key] = variance
            save_dict["Var[y|x,D]"] = total_variance
            save_dict["Va_variance"] = Va_variance
            save_dict["Ve_variance"] = Ve_variance
            save_dict["E[y|x,D]"] = mean_y
            save_dict["kl_pyx_pyxz"] = kl_pyx_pyxz
            save_dict["kl_pyxz_pyx"] = kl_pyxz_pyx
            save_dict["api_calls"] = self.num_api_calls
            
            save_dict_list.append(save_dict)
            
        save_df = pd.DataFrame(save_dict_list)
                
        save_df = calculate_min_Va_by_KL_rank(save_df, self.config.min_KL_rank, upper_bound_by_total_U=True, uncertainty_type="variance")
        
        return save_df
            
    def get_single_trial_action(self, trial: int, context: Optional[pd.Series] = None, random_action: bool = False, uncertainty_type: str = "epistemic"):
        if random_action:
            action_taken = self.get_random_action()
        else:
            Q_values = {}
            U_values = {}
            UCB_values = {}
            for action in tqdm(self.action_space):
                save_df = self.process_single_trial_action(trial=trial, action=action, context=context)
                
                Q_values.update({action: save_df["E[y|x,D]"].values[0]})
                if uncertainty_type == "epistemic":
                    U_values.update({action: np.sqrt(save_df["max_Ve_variance"].values[0])})
                elif uncertainty_type == "total":
                    U_values.update({action: np.sqrt(save_df["Var[y|x,D]"].values[0])})
                elif uncertainty_type == "ucb1":
                    UCB_uncertainty = self.UCB1_algorithm.get_uncertainty(action)
                    U_values.update({action: UCB_uncertainty})
                
                UCB_values.update({action: Q_values[action] + self.config.bandit_exploration_rate * U_values[action]})
                            
            print(f"Q values: {Q_values}")
            print(f"U values: {U_values}")
            print(f"UCB values: {UCB_values}")
            
            self.Q_values = Q_values
            self.U_values = U_values
            self.UCB_values = UCB_values
            
            max_UCB_value = max(UCB_values.values())
            max_UCB_action = [action for action, value in UCB_values.items() if value == max_UCB_value]
            action_taken = self.rng.choice(max_UCB_action)
        return action_taken
    
    def single_trial(self, trial: int):
        print(f"\nTrial {trial + 1}/{self.config.num_trials}")
        
        if self.config.is_contextual_bandit:
            context = self.bandit.get_next_context()
        else:
            context = None
            
        is_random_action = trial < self.config.num_random_trials
        action_taken = self.get_single_trial_action(trial, context, is_random_action, self.config.uncertainty_type)
        
        reward = self.bandit.get_reward(action_taken)
        
        regret = self.bandit.get_optimal_mean_reward() - reward

        print(f"Action: {action_taken}; Reward: {reward}; Regret: {regret}")
        
        if self.config.uncertainty_type == "ucb1":
            self.UCB1_algorithm.update(action_taken, reward)
            
        if isinstance(context, pd.Series | pd.DataFrame):
            trial_df = context
        else:
            if self.config.is_contextual_bandit:
                trial_df = pd.DataFrame({key: [value] for key, value in context.items()})
                trial_df["note"] = BanditClassificationUtils.parse_features_and_action_to_note(action=action_taken, row=context, feature_columns = self.feature_columns, decimal_places=self.config.decimal_places)
            else:
                trial_df = pd.DataFrame({"note": [BanditClassificationUtils.parse_features_and_action_to_note(action_taken, decimal_places=self.config.decimal_places)]})
                
        trial_df["action"] = action_taken
        trial_df["label"] = reward
        trial_df["regret"] = regret
        
        trial_df["trial"] = trial
        trial_df["optimal_action"] = self.bandit.optimal_action()
        
        if not is_random_action:
            for action in self.action_space:
                trial_df[f"Q_value_{action}"] = self.Q_values[action]
            for action in self.action_space:
                trial_df[f"U_value_{action}"] = self.U_values[action]
            for action in self.action_space:
                trial_df[f"UCB_value_{action}"] = self.UCB_values[action]
          
        if trial == 0:
            self.D_rows = trial_df
        else:
            self.D_rows = pd.concat([self.D_rows, trial_df], ignore_index=True)
            
        self.save_D_rows()
    
    
    def run_experiment(self):
        for trial in range(self.config.num_trials):
            self.single_trial(trial)
            
        print(f"\nTotal API calls: {self.num_api_calls}")
        
def main():
    config = BanditClassificationExperimentConfig(**vars(args))
    
    experiment = BanditClassificationExperiment(config)
    
    experiment.run_experiment()

if __name__ == "__main__":
    main()