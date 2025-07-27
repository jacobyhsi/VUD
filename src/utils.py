import math
import re
import ast
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Optional
from itertools import product
from src.chat import chat_perturb
from tqdm import tqdm

# Helper Functions

def calculate_entropy(probs: dict):
    # Calculate entropy using all probabilities in the dictionary
    probs_array = np.array(list(probs.values()))
    entropy = -np.sum(probs_array * np.log2(probs_array))
    return round(entropy, 5)

def calculate_discrete_mean(probs: dict):
    # Calculate mean
    cum_mean = 0
    for label, prob in probs.items():
        y_value = float(int(label))
        cum_mean += y_value * prob
    return round(cum_mean, 5)

def calculate_discrete_variance(probs: dict):
    # Calculate variance
    cum_mean = 0
    cum_sum_squared = 0
    for label, prob in probs.items():
        y_value = float(int(label))
        cum_mean += y_value * prob
        cum_sum_squared += (y_value ** 2) * prob
    variance = cum_sum_squared - (cum_mean ** 2)
    return round(variance, 5)

def calculate_kl_divergence(p, q):
    epsilon = 1e-12  # small constant to avoid log(0)
    kl = 0.0
    for label in p:
        p_val = p[label] + epsilon
        q_val = q[label] + epsilon
        kl += p_val * (np.log(p_val) - np.log(q_val))
    return kl

def calculate_kl_divergence_for_z_data(df: pd.DataFrame):
    kl_divergence_pyx_pyxz = []
    kl_divergence_pyxz_pyx = []

    PROB_LABELS = ["0", "1"]
    for index, row in df.iterrows():
        prob_y_x = {}
        prob_y_xz = {}
        for label in PROB_LABELS:
            prob_y_x[label] = row[f"p(y={label}|x,D)"]
            prob_y_xz[label] = row[f"p(y={label}|x,z,D)"]
        kl_divergence_pyx_pyxz.append(calculate_kl_divergence(prob_y_x, prob_y_xz))
        kl_divergence_pyxz_pyx.append(calculate_kl_divergence(prob_y_xz, prob_y_x))
        
    df["kl_pyx_pyxz"] = kl_divergence_pyx_pyxz
    df["kl_pyxz_pyx"] = kl_divergence_pyxz_pyx
    
    return df

def calculate_min_Va_by_KL_threshold(save_data: pd.DataFrame, threshold: float = 0.01, forward_kl = True, uncertainty_type: str = "entropic"):
    valid_Va = []
    if uncertainty_type == "entropic":
        total_U = save_data["H[p(y|x,D)]"][0]
        aleatoric_key = "Va"
        epistemic_key = "Ve"
    elif uncertainty_type == "variance":
        total_U = save_data["Var[y|x,D]"][0]
        aleatoric_key = "Va_variance"
        epistemic_key = "Ve_variance"
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}. Choose either 'entropic' or 'variance'.")
    for i, row in save_data.iterrows():
        if forward_kl:
            if row["kl_pyx_pyxz"] <= threshold:
                valid_Va.append(row[aleatoric_key])
        else:
            if row["kl_pyxz_pyx"] <= threshold:
                valid_Va.append(row[aleatoric_key])
    if len(valid_Va) == 0:
        min_Va = np.nan
        save_data[f"within_threshold"] = False
        save_data[f"z_value_for_min_{aleatoric_key}"] = False
    else:
        min_Va = min(valid_Va)
        save_data[f"within_threshold"] = save_data[aleatoric_key].apply(lambda x: x in valid_Va)
        save_data[f"z_value_for_min_{aleatoric_key}"] = save_data[aleatoric_key].apply(lambda x: x == min_Va)
    save_data[f"min_{aleatoric_key}"] = min_Va
    max_Ve = round(total_U - min_Va, 5)
    if min_Va == np.nan:
        save_data[f"max_{epistemic_key}"] = np.nan
    else:
        save_data[f"max_{epistemic_key}"] = max_Ve
    
    return save_data

def calculate_min_Va_by_KL_rank(save_data: pd.DataFrame, num_valid_Va: int = 5, forward_kl = True, upper_bound_by_total_U = False, uncertainty_type: str = "entropic"):
    if uncertainty_type == "entropic":
        total_U = save_data["H[p(y|x,D)]"][0]
        aleatoric_key = "Va"
        epistemic_key = "Ve"
    elif uncertainty_type == "variance":
        total_U = save_data["Var[y|x,D]"][0]
        aleatoric_key = "Va_variance"
        epistemic_key = "Ve_variance"
    else:
        raise ValueError(f"Invalid uncertainty type: {uncertainty_type}. Choose either 'entropic' or 'variance'.")
    if forward_kl:
        kl_values = save_data["kl_pyx_pyxz"]
    else:
        kl_values = save_data["kl_pyxz_pyx"]
    # min kl values
    min_kl_values = kl_values.nsmallest(num_valid_Va)
    save_data["within_threshold"] = kl_values.isin(min_kl_values)
    min_Va = save_data[save_data["within_threshold"]][aleatoric_key].min()
    save_data["z_value_for_min_Va"] = save_data[aleatoric_key].apply(lambda x: x == min_Va)
    if upper_bound_by_total_U:
        min_Va = min(min_Va, total_U)
    save_data[f"min_{aleatoric_key}"] = min_Va
    max_Ve = round(total_U - min_Va, 5)
    save_data[f"max_{epistemic_key}"] = max_Ve
    
    return save_data

def extract(text, tag_text: str = "output"):
    match = re.search(fr'(.*?)</{tag_text}>', text, re.DOTALL | re.IGNORECASE)
    if match:
        output_str = match.group(1).strip()
        output_dict = ast.literal_eval(output_str)
        return output_dict
    else:
        print("Could not find output tags in the response.")
        raise ValueError("Invalid response format.")
    
class ToyDataUtils:
    @staticmethod
    def parse_features_to_note(row: pd.Series | pd.DataFrame, feature_columns: list[str]):
        note_parts = []
        for feature in feature_columns:
            note_parts.append(f"{feature} = {row[feature]}")
        # join note with ;
        return "; ".join(note_parts)
    
    @staticmethod
    def get_feature_columns(data: pd.DataFrame):
        return [col for col in data.columns if col not in ['note', 'label']]
    
    @staticmethod
    def create_x_row_from_x_features(x_features: str, feature_columns: list[str], **kwargs):
        """ 
        Create x_row from given x_features.
        
        x_features is a string with the format "{'feature1': [f1_1, ..., f1_n], 'feature2': [f2_1, ..., f2_n], ...}"
        """
        x_features = ast.literal_eval(x_features)
        x_row = pd.DataFrame(x_features)
        x_row["label"] = 0
        x_row["note"] = x_row.apply(
            lambda row: ToyDataUtils.parse_features_to_note(row, feature_columns),
            axis=1,
        )
                
        return x_row

    @staticmethod
    def create_x_row_from_x_range(x_range: str, feature_columns: list[str], decimal_places: int, **kwargs):
        """
        Create x_row grid for a given x_range.
        
        x_range is a string with the format "start, end, step" for each feature.
        
        Example:
        x_range = "{'x1': [0, 10, 0.2],'x2': [1, 5, 1]}"
        """
        
        x_range_dict = ast.literal_eval(x_range)
        x_row = pd.DataFrame()
        
        for feature, (start, end, step) in x_range_dict.items():
            x_range_dict[feature] = np.round(np.arange(float(start), float(end), float(step)), decimal_places)

        values = product(*x_range_dict.values())
        x_row = pd.DataFrame(values, columns=x_range_dict.keys())
        
        x_row["label"] = 0
        x_row["note"] = x_row.apply(
            lambda row: ToyDataUtils.parse_features_to_note(row, feature_columns),
            axis=1
        )
                
        return x_row

    @staticmethod
    def create_x_row_from_test_data(
        test_data: pd.DataFrame,
        num_x_samples: int,
        x_sample_seed: int,
        **kwargs,
    ):
        x_row = test_data.sample(n=num_x_samples, random_state=x_sample_seed)
        
        return x_row
    
    @staticmethod
    def create_x_row(method_name: str, **kwargs):
        if method_name == "x_features":
            return ToyDataUtils.create_x_row_from_x_features(**kwargs)
        elif method_name == "x_range":
            return ToyDataUtils.create_x_row_from_x_range(**kwargs)
        elif method_name == "sample":
            return ToyDataUtils.create_x_row_from_test_data(**kwargs)
        else:
            raise ValueError(f"Invalid method_name: {method_name}")
        
    @staticmethod
    def create_icl_data(num_shots: int, data: pd.DataFrame, icl_sample_seed: int):
        pass

class ToyClassificationUtils(ToyDataUtils):
    pass

class GaussianDistribution:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
        
    @property
    def entropy(self):
        return 0.5 * np.log(2 * np.pi * self.std**2) + 0.5
    
    def pdf(self, x: float):
        return stats.norm.pdf(x, loc=self.mean, scale=self.std)
    
    def sample(self, size: Optional[int] = None):
        return np.random.normal(loc=self.mean, scale=self.std, size=size)

class ToyRegressionUtils(ToyDataUtils):
    @staticmethod
    def gaussian_from_samples(data: list[float], num_outlier_pairs_to_remove: int = 0, std_method: str = "default"):
        """ 
        Generate a Gaussian distribution from the given data.
        
        If num_outlier_pairs_to_remove is greater than 0, it will remove the specified number of outlier pairs from both ends of the data. To compute trimmed mean
        
        If iqr_scale_estimator is True, it will use the IQR method to estimate the standard deviation. Otherwise, use trimmed standard deviation.
        """
        if num_outlier_pairs_to_remove > 0:
            sorted_data = sorted(data)[num_outlier_pairs_to_remove:-num_outlier_pairs_to_remove]
        else:
            sorted_data = sorted(data)
        mean = np.mean(sorted_data)
        if std_method == "iqr":
            std = 0.5 * stats.iqr(data, nan_policy="omit") / stats.norm.ppf(0.75)
        elif std_method == "default":
            std = np.std(sorted_data) * np.sqrt(len(sorted_data) / (len(sorted_data) - 1))
        else:
            raise ValueError(f"Invalid std_method: {std_method}")
        return GaussianDistribution(mean, std)
    
    @staticmethod
    def calculate_kl_divergence(p: GaussianDistribution, q: GaussianDistribution):
        kl = np.log(q.std / p.std) + (p.std**2 + (p.mean - q.mean)**2) / (2 * q.std**2) - 0.5
        return kl
    
class BanditUtils:
    @staticmethod
    def parse_features_and_action_to_note(action: int|str, row: Optional[pd.Series] = None, feature_columns: list[str]=[], decimal_places: int = 2):
        note_parts = []
        for feature in feature_columns:
            if isinstance(row[feature], float):
                note_parts.append(f"{feature} = {row[feature]:.{decimal_places}f}")
            else:
                note_parts.append(f"{feature} = {row[feature]}")
        note_parts.append(f"action = {action}")
        # join note with ;
        return "; ".join(note_parts)
    
    @staticmethod
    def get_feature_columns(data: pd.DataFrame):
        return [col for col in data.columns if col not in ['note', 'label']]
    
class BanditClassificationUtils(BanditUtils):
    pass

class QAUtils:
    def perturb_z(
        data: pd.DataFrame,
        x_row: pd.Series,
        z_samples: int,
        seed: int,
        dataname: str,
        max_tokens: int = 512,
    ) -> pd.DataFrame:
        replace_flag = z_samples > len(data)
        z_base = data.sample(n=z_samples, replace=replace_flag, random_state=seed).reset_index(drop=True)

        perturbed_rows = []
        for idx, z_row in tqdm(z_base.iterrows(), total=len(z_base), desc="Processing z perturbations"):

            if dataname == "mmlu":
                prompt = (
                    "/nothink Please rephrase the following:\n\n"
                    f"{z_row['note']}\n\n"
                    "While rephrasing the above, you must incorporate context from the following and make sure it's intertwined/interconnected:\n\n"
                    f"{x_row['note']}\n\n"
                    "Use the following format when rephrasing:\n\n"
                    "<rep> Question:... Choices... </rep>"
                )
            else:
                prompt = (
                    "Please rephrase the following:\n\n"
                    f"{z_row['note']}\n\n"
                    "While rephrasing the above, incorporate context from the following and make sure it's intertwined/interconnected:\n\n"
                    f"{x_row['note']}\n\n"
                    "Use the following format when rephrasing:\n\n"
                    "<rep> Question: {{Rephrased Question}}? Context: {{Rephrased Context}}. </rep>"
                )

            MAX_RETRIES = 3
            note = z_row["note"].lower()  # fallback

            for attempt in range(MAX_RETRIES):
                attempt_seed = seed + idx * MAX_RETRIES + attempt  # unique seed per z + retry
                try:
                    response = chat_perturb(prompt, seed=attempt_seed, max_tokens=max_tokens)
                    match = re.search(r"<rep>(.*?)</rep>", response, flags=re.DOTALL | re.IGNORECASE)

                    if match:
                        note = match.group(1).strip().lower()
                        break  # success
                    else:
                        print(f"[Retry {attempt+1}] <rep> tags not found. Retrying...")
                        print("Response that failed:")
                        print(response)

                except Exception as e:
                    print(f"[Retry {attempt+1}] chat_perturb failed: {e}")

            perturbed_rows.append({"note": note, "label": z_row["label"]})

        return pd.DataFrame(perturbed_rows)
