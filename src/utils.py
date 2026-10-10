import math
import os
import re
import ast
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Optional
from itertools import product
from src.chat import chat_perturb
from tqdm import tqdm

# Helper Functions

def calculate_entropy(probs: dict):
    probs_array = np.asarray(list(probs.values()), dtype=float)
    if np.any(probs_array < 0):
        raise ValueError("Probabilities must be non-negative.")
    positive_probs = probs_array[probs_array > 0]
    entropy = -np.sum(positive_probs * np.log2(positive_probs))
    return round(float(entropy), 5)

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


_LEGACY_X_CSV = re.compile(r"_x(\d+)\.csv$")


def split_result_x_frames(save_data: pd.DataFrame) -> list[pd.DataFrame]:
    """One frame per query x. A combined results CSV holds many x values; a legacy file holds one."""
    if save_data.empty:
        return []
    if "x_note" not in save_data.columns or save_data["x_note"].nunique(dropna=False) <= 1:
        return [save_data.reset_index(drop=True)]
    return [group.reset_index(drop=True) for _, group in save_data.groupby("x_note", sort=False)]


def read_results_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        text = path.read_text()
        newline = text.rfind("\n")
        if newline <= 0:
            raise
        return pd.read_csv(StringIO(text[: newline + 1]))


def load_result_x_frames(output_dir: str) -> list[pd.DataFrame]:
    """Load every results CSV in a directory as one frame per x. Combined rows win over a legacy per-x file."""
    directory = Path(output_dir)
    if not directory.is_dir():
        return []
    paths = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix == ".csv" and path.name.startswith("results_") and path.stat().st_size > 0
    ]
    paths.sort(key=lambda path: (_LEGACY_X_CSV.search(path.name) is not None, path.name))
    frames = []
    seen = set()
    for path in paths:
        for frame in split_result_x_frames(read_results_csv(path)):
            if "x_note" not in frame.columns or frame.empty:
                frames.append(frame)
                continue
            note = frame["x_note"].iloc[0]
            if note in seen:
                continue
            seen.add(note)
            frames.append(frame)
    return frames


def combined_results_path(output_dir: str, run_name: str) -> Path:
    return Path(output_dir) / f"results_{run_name}.csv"


def append_x_results(output_dir: str, run_name: str, save_df: pd.DataFrame) -> Path:
    path = combined_results_path(output_dir, run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    if not write_header:
        existing = pd.read_csv(path, nrows=0).columns.tolist()
        if existing != list(save_df.columns):
            raise ValueError(
                f"Column mismatch appending to {path}. Start a new --save_directory instead of mixing schemas."
            )
    with path.open("a", newline="") as handle:
        save_df.to_csv(handle, header=write_header, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _drop_incomplete_x_tail(save_data: pd.DataFrame, num_z: int) -> tuple[pd.DataFrame, int]:
    if save_data.empty or "x_note" not in save_data.columns or num_z <= 0:
        return save_data, 0
    last_note = save_data["x_note"].iloc[-1]
    is_last = save_data["x_note"] == last_note
    last_count = int(is_last.sum())
    if last_count == num_z or last_count > num_z:
        return save_data, 0
    if not bool(is_last.iloc[-last_count:].all()):
        return save_data, 0
    return save_data.iloc[:-last_count].copy(), last_count


def resume_completed_x(
    output_dir: str,
    run_name: str,
    x_notes: list,
    num_z: int,
    x_save_value: int,
) -> tuple[set[int], Optional[int], int]:
    """x indices already on disk, api_calls from the latest complete slice, and rows dropped from a partial tail."""
    directory = Path(output_dir)
    done: set[int] = set()
    api_calls = None
    best_idx = -1
    dropped = 0
    note_to_idx = {}
    for idx, note in enumerate(x_notes):
        note_to_idx.setdefault(str(note), idx)

    def consider(idx: int, frame: pd.DataFrame) -> None:
        nonlocal api_calls, best_idx
        if idx > best_idx and "api_calls" in frame.columns and len(frame):
            best_idx = idx
            api_calls = int(frame["api_calls"].iloc[-1])

    combined = combined_results_path(output_dir, run_name)
    if combined.is_file() and combined.stat().st_size > 0:
        try:
            saved = pd.read_csv(combined)
            torn = False
        except pd.errors.ParserError:
            saved = read_results_csv(combined)
            torn = True
        trimmed, dropped = _drop_incomplete_x_tail(saved, num_z)
        if torn or dropped:
            if trimmed.empty:
                combined.unlink()
            else:
                temporary = combined.with_name(combined.name + ".tmp")
                trimmed.to_csv(temporary, index=False)
                os.replace(temporary, combined)
            saved = trimmed
        if not saved.empty and "x_note" in saved.columns:
            for note, group in saved.groupby("x_note", sort=False):
                idx = note_to_idx.get(str(note))
                if idx is None:
                    continue
                done.add(idx)
                consider(idx, group)

    legacy_pattern = re.compile(rf"^results_{re.escape(str(run_name))}_x(\d+)\.csv$")
    legacy = []
    if directory.is_dir():
        for path in directory.iterdir():
            match = legacy_pattern.fullmatch(path.name)
            if match is None or not path.is_file() or path.stat().st_size == 0:
                continue
            idx = int(match.group(1)) - x_save_value
            if idx < 0 or idx >= len(x_notes) or idx in done:
                continue
            legacy.append((idx, path))
    for idx, _path in legacy:
        done.add(idx)
    if legacy:
        idx, path = max(legacy)
        if idx > best_idx:
            consider(idx, read_results_csv(path))
    return done, api_calls, dropped

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

    def logpdf(self, y: float) -> float:
        if not np.isfinite(self.std) or self.std <= 0:
            return float("-inf")
        return float(stats.norm.logpdf(y, loc=self.mean, scale=self.std))
    
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
        model: str = "Qwen/Qwen2.5-14B",
        host: str = "localhost",
        port: int = 8000,
    ) -> pd.DataFrame:
        replace_flag = z_samples > len(data)
        z_base = data.sample(n=z_samples, replace=replace_flag, random_state=seed).reset_index(drop=True)

        perturbed_rows = []
        for idx, z_row in tqdm(z_base.iterrows(), total=len(z_base), desc="Processing z perturbations"):

            if dataname in {"mmlu", "mmlu_cs", "mmlu_moral"}:
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
                    response = chat_perturb(
                        prompt,
                        seed=attempt_seed,
                        max_tokens=max_tokens,
                        model=model,
                        ip=host,
                        port=port,
                    )
                    match = re.search(r"<rep>(.*?)</rep>", response, flags=re.DOTALL | re.IGNORECASE)
                    if match is None:
                        # Diffusion often hits max_new_tokens before </rep>.
                        match = re.search(r"<rep>(.*)", response, flags=re.DOTALL | re.IGNORECASE)

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
