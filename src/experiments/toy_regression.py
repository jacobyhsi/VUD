import os
import re
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass

from src.dataset import load_dataset
from src.bayesian_optimisation import new_candidate
from src.utils import ToyRegressionUtils, GaussianDistribution, extract, calculate_min_Va_by_KL_rank
from src.prompt import ToyRegressionPrompt
from src.chat import chat_response_only
from src.parallel.va_estimators import get_regression_va_estimator
from src.parallel.regression_parallel_stats import (
    DEFAULT_PP_COVARIANCE_FLOOR,
    PP_GAUSSIAN_ESTIMATOR,
)

pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser(description='Running Toy Classification')

"""LLM API Configuration"""
parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B", type=str)
parser.add_argument("--model_port", default="8000", type=str)
parser.add_argument("--model_ip", default="localhost", type=str)
parser.add_argument("--model_temperature", default=1, type=float)
parser.add_argument("--is_local_client", default=1, type=int)

"""Dataset Configuration"""
parser.add_argument("--dataset_name", default="linear_regression", type=str)
parser.add_argument("--D_size", default=15, type=int)

"""X Configuration"""
parser.add_argument("--x_row_method", type=str, default="x_range")
parser.add_argument("--num_x_samples", default=1, type=int)
parser.add_argument("--x_features", default=None)
parser.add_argument("--x_range", default="{'x1': [-15, 15, 0.2]}")
parser.add_argument("--x_sample_seed", default=0, type=int)
parser.add_argument("--decimal_places", default=1, type=int)

"""Permutation Related Configuration"""
parser.add_argument("--num_permutations", default=10, type=int)
parser.add_argument("--permute_context", default=1, type=int)

"""Seed Configuration"""
parser.add_argument("--numpy_seed", default=0, type=int)
parser.add_argument("--data_split_seed", default=0, type=int)
parser.add_argument("--icl_sample_seed", default=0, type=int)
parser.add_argument("--fixed_permutation_seed", default=0, type=int)

"""Z Configuration"""
parser.add_argument("--num_z", default=5, type=int)
parser.add_argument("--perturb_about_x", default=1, type=int)
parser.add_argument("--perturbation_std", default=0.1, type=float)
parser.add_argument("--num_bo_z", default=0, type=int) # Bayesian Optimisation Params
parser.add_argument("--num_candidates", default=3, type=int) # Bayesian Optimisation Params

"""Distribution Approximation Configuration"""
parser.add_argument("--num_outlier_pairs_to_remove", default=0, type=int)
parser.add_argument("--std_method", default="default", type=str)

"""Save Configuration"""
parser.add_argument("--run_name", default="test")
parser.add_argument("--save_directory", default=None, help="Defaults to --model_name.")
parser.add_argument("--x_save_value", default=0, type=int)
parser.add_argument("--num_api_calls_save_value", default=0, type=int)

parser.add_argument("--verbose_output", default=0, type=int)
parser.add_argument("--plot", default=1, type=int, help="Save the 1D total-uncertainty decomposition plot during the run. Default 1.")
parser.add_argument(
    "--resume",
    default=0,
    type=int,
    help="Skip x-slices whose result CSV already exists and continue from the first missing slice.",
)
parser.add_argument(
    "--va_method",
    default="vanilla",
    type=str,
    choices=("vanilla", "parallel_pf", "parallel_pp", "parallel", "parallel_hybrid"),
    help=(
        "Va estimator: vanilla (forward factorization), parallel_pf (joint-canvas "
        "samples scored by the forward conditional Gaussian), parallel_pp (joint-Gaussian "
        "conditional entropy with no forward scoring), parallel (legacy PF alias), or "
        "parallel_hybrid (parallel u with conditional entropy of sequential y|u)."
    ),
)
parser.add_argument(
    "--num_joint_samples",
    default=10,
    type=int,
    help="Number of paired (u, y) denoising samples used by parallel regression.",
)
parser.add_argument(
    "--num_masks_per_slot",
    default=8,
    type=int,
    help="Fallback mask tokens per numeric slot if D-label inference is unavailable.",
)
parser.add_argument(
    "--pp_covariance_floor", type=float, default=DEFAULT_PP_COVARIANCE_FLOOR,
    help="Regression PP only: absolute eigenvalue floor for the fitted joint covariance "
         "(output-squared units). Default 1e-6; 0 rejects singular fits. Saved in results.",
)
args = None


def parse_args(argv=None):
    global args
    args = parser.parse_args(argv)
    save_directory_was_default = args.save_directory is None
    if args.save_directory is None:
        args.save_directory = args.model_name.rsplit("/", 1)[-1]
    if args.va_method != "vanilla" and save_directory_was_default:
        args.save_directory = f"{str(args.save_directory).rstrip('/')}/va_{args.va_method}"
    save_dir = str(args.save_directory).rstrip("/")
    seed_dir = f"seed_{args.icl_sample_seed}"
    if not save_dir.endswith(seed_dir):
        args.save_directory = f"{save_dir}/{seed_dir}"
    return args

@dataclass
class ToyRegressionExperimentConfig:
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
    
    num_outlier_pairs_to_remove: int
    std_method: str
    
    run_name: int
    save_directory: int
    x_save_value: int
    num_api_calls_save_value: int
    
    verbose_output: int
    plot: int
    resume: int
    va_method: str
    num_joint_samples: int
    num_masks_per_slot: int
    pp_covariance_floor: float = DEFAULT_PP_COVARIANCE_FLOOR


def pp_run_config_json(config):
    """Record settings that affect sampling/fitting; allow display/resume changes."""
    ignored = {"resume", "plot", "verbose_output", "save_directory", "num_api_calls_save_value"}
    return json.dumps({k: v for k, v in vars(config).items() if k not in ignored}, sort_keys=True)


def validate_pp_result_directory(output_dir, config):
    """Reject mixed PP results before even overwriting the saved demonstrations."""
    current_pp = config.va_method == "parallel_pp"
    for path in sorted(Path(output_dir).glob("results_*.csv")):
        if path.stat().st_size == 0:
            continue
        rows = pd.read_csv(path)
        saved_pp = "va_method" in rows and rows["va_method"].eq("parallel_pp").any()
        if not current_pp and not saved_pp:
            continue
        if not current_pp or rows.empty or "va_method" not in rows or not rows["va_method"].eq("parallel_pp").all():
            raise ValueError(f"Cannot mix parallel_pp with other regression results in {path.parent}; "
                             "use a separate --save_directory")
        expected = {
            "parallel_joint_estimator": PP_GAUSSIAN_ESTIMATOR,
            "parallel_pp_run_config_json": pp_run_config_json(config),
        }
        for key, value in expected.items():
            if key not in rows or not rows[key].eq(value).all():
                raise ValueError(f"Parallel PP settings/estimator mismatch in {path} ({key}); "
                                 "use a separate --save_directory")

def save_toy_regression_plots(output_dir: str, run_name: str) -> list[str]:
    """Save the total-uncertainty plot using the code in eval/eval_toy_1d_reg.ipynb."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from csaps import csaps

    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from notebook_utils import set_plot_style

    set_plot_style()

    results_directory = output_dir if output_dir.endswith(os.sep) else output_dir + os.sep

    D_data = None
    for filename in os.listdir(results_directory):
        if filename.startswith("D_") and filename.endswith(".csv"):
            D_data = pd.read_csv(results_directory + filename)
            break

    if D_data is None:
        print(f"Skipping plot: missing ICL file in {output_dir}")
        return []

    feature_column = ToyRegressionUtils.get_feature_columns(D_data)[0]
    x_col = f"x_{feature_column}"

    df_list = []
    for filename in os.listdir(results_directory):
        if filename.startswith("results_") and filename.endswith(".csv"):
            df_list.append(pd.read_csv(results_directory + filename))

    if not df_list:
        print(f"Skipping plot: no results_*.csv files in {output_dir}")
        return []

    rows = []
    for z_df in df_list:
        z_df = calculate_min_Va_by_KL_rank(z_df, num_valid_Va=5, forward_kl=True, upper_bound_by_total_U=True)
        try:
            rows.append({
                x_col: float(z_df[x_col].values[0]),
                "total_uncertainty": float(z_df["H[p(y|x,D)]"].values[0]),
                "min_Va": float(z_df["min_Va"].values[0]),
            })
        except Exception:
            continue

    if not rows:
        print(f"Skipping plot: no valid result rows in {output_dir}")
        return []

    results_df = pd.DataFrame(rows).sort_values(by=x_col)
    valid_h = np.isfinite(results_df["total_uncertainty"])
    valid_va = np.isfinite(results_df["min_Va"])
    if valid_h.sum() < 1:
        print(f"Skipping plot: no finite H[p(y|x,D)] in {output_dir}")
        return []

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    plt.scatter(
        results_df.loc[valid_h, x_col],
        results_df.loc[valid_h, "total_uncertainty"],
        color="C0",
        alpha=0.4,
    )
    if valid_va.any():
        plt.scatter(
            results_df.loc[valid_va, x_col],
            results_df.loc[valid_va, "min_Va"],
            color="C1",
            alpha=0.4,
        )

    x_min = float(results_df[x_col].min())
    x_max = float(results_df[x_col].max())
    if x_max > x_min:
        x_grid = np.linspace(x_min, x_max, 100)
        if valid_h.sum() >= 2:
            y_total = csaps(
                results_df.loc[valid_h, x_col],
                results_df.loc[valid_h, "total_uncertainty"],
                smooth=0.85,
            )
            plt.plot(x_grid, y_total(x_grid), color="C0", linewidth=3, label="Total Uncertainty")
        if valid_va.sum() >= 2:
            y_min_va = csaps(
                results_df.loc[valid_va, x_col],
                results_df.loc[valid_va, "min_Va"],
                smooth=0.85,
            )
            plt.plot(x_grid, y_min_va(x_grid), color="C1", linewidth=3, label="Aleatoric Uncertainty")
    elif valid_h.any():
        plt.plot([], [], color="C0", linewidth=3, label="Total Uncertainty")

    label_seen = False
    for _, row in D_data.iterrows():
        label_string = "ICL Data" if not label_seen else None
        label_seen = True
        plt.axvline(x=row[feature_column], color="C5", linestyle="--", alpha=0.5, linewidth=3, label=label_string)

    plt.title("Total Uncertainty Decomposition: Toy Regression")
    plt.ylabel("Uncertainty")
    plt.xlabel(r"Test Covariate $x$")
    plt.legend(framealpha=0.75)

    plot_path = os.path.join(output_dir, f"plot_entropy_decomposition_{run_name}.png")
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [plot_path]

class ToyRegressionExperiment:
    def __init__(self, config: ToyRegressionExperimentConfig):
        self.config = config
        
        np.random.seed(self.config.numpy_seed)

        self.prompter = ToyRegressionPrompt(model_name=self.config.model_name)
        
        if self.config.num_bo_z > self.config.num_z:
            raise ValueError("Number of bo z values cannot be greater than number of z values.")

        self.va_estimator = get_regression_va_estimator(
            self.config.va_method,
            self.config.num_joint_samples,
            self.config.num_masks_per_slot,
            self.config.pp_covariance_floor,
        )
        output_dir = f"results/toy_regression/{self.config.dataset_name}/{self.config.save_directory}"
        validate_pp_result_directory(output_dir, self.config)
        self.data_preprocessing()
        self.num_api_calls = self.config.num_api_calls_save_value
        print(f"Va estimator: {self.va_estimator.name}")

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

        D_rows = data.sample(n=self.config.D_size, random_state=self.config.icl_sample_seed)
        self.D_feature_means = D_rows[self.feature_columns].mean().to_numpy()
        self.D_feature_stds = D_rows[self.feature_columns].std().to_numpy()

        self.D_note_label_df = D_rows[['note', 'label']]

        output_dir = f"results/toy_regression/{self.config.dataset_name}/{self.config.save_directory}"
        os.makedirs(output_dir, exist_ok=True)
        D_rows.to_csv(f"{output_dir}/D_{self.config.run_name}.csv", index=False)
        
        self.max_D_label = D_rows['label'].max()
        self.min_D_label = D_rows['label'].min()
    
    def get_next_z(self, z_idx: int, x_idx: int):
        if z_idx < self.config.num_z - self.config.num_bo_z:
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
                self.z_data["note"] = self.z_data.apply(lambda row: ToyRegressionUtils.parse_features_to_note(row, self.feature_columns), axis=1)
                
            else:
                modified_row = self.z_data.loc[z_idx-1].copy()
                modified_row[self.feature_columns] = new_value
                modified_row["note"] = ToyRegressionUtils.parse_features_to_note(modified_row, self.feature_columns)
                
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
                    permutation_seed=permutation_seed if self.config.permute_context else self.config.fixed_permutation_seed,
                    icl_z_note=icl_z_note,
                    icl_u_label=icl_u_label,
                )
            
                if self.config.verbose_output:
                    print(f"Prompt for {probability_calculated}:")
                    print(prompt)

                # Get the prediction and probabilities from the model
                response = chat_response_only(prompt, seed=permutation_seed, model=self.config.model_name, port=self.config.model_port, ip=self.config.model_ip, temperature=self.config.model_temperature, is_local_client=self.config.is_local_client)
                
                self.num_api_calls += 1     
                attempts += 1        

                try:
                    sample = extract(response)
                except ValueError:
                    # Diffusion models often omit </output>; accept a bare number.
                    sample = float(response.strip())
                
                if not isinstance(sample, float|int):
                    print(f"Invalid sample for {probability_calculated}: {sample}")
                    raise ValueError(f"Invalid sample for {probability_calculated}: {sample}")
                
                if self.config.verbose_output:
                    print(f"y_sample: {sample}")
                
                distribution_samples.append(sample)

                successful_seeds += 1
                
            except Exception as exc:
                print(
                    f"Call {self.num_api_calls} failed: "
                    f"{type(exc).__name__}: {exc}. "
                    f"Restarting for seed {successful_seeds}"
                )
                        
        if successful_seeds == 0:
            raise ValueError(f"All seeds failed for {probability_calculated}.")      
        
        gaussian = ToyRegressionUtils.gaussian_from_samples(distribution_samples, self.config.num_outlier_pairs_to_remove, self.config.std_method)
        
        if self.config.verbose_output:
            print(f"\nGaussian Approximation for {probability_calculated}: mean = {gaussian.mean}, std = {gaussian.std}")
            
        return gaussian, distribution_samples
            
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
            
        for i in tqdm(range(self.config.num_z)):

            self.get_next_z(i, x_idx)
            
            row = self.z_data.iloc[i]
            
            z = row['note']

            z_fields = self.va_estimator.estimate(
                self, x, z, pyx_gaussian, Hyx, total_variance
            )
            self.z_BO_maximisation_objective.append(-z_fields["Va"] - z_fields["kl_pyx_pyxz"])

            save_dict = {f"z_{feature}": row[feature] for feature in self.feature_columns}
            save_dict["z_note"] = z
            save_dict_x = {f"x_{feature}": self.x_row.iloc[x_idx][feature] for feature in self.feature_columns}
            save_dict_x["x_note"] = x
            save_dict = {**save_dict, **save_dict_x}
            save_dict["p(y|x,D)_mean"] = pyx_gaussian.mean
            save_dict["p(y|x,D)_std"] = pyx_gaussian.std
            save_dict["H[p(y|x,D)]"] = Hyx
            save_dict["Var[y|x,D]"] = total_variance
            save_dict.update(z_fields)
            if self.va_estimator.name == "parallel_pp":
                save_dict["parallel_pp_run_config_json"] = pp_run_config_json(self.config)
            save_dict["api_calls"] = self.num_api_calls
            save_dict_list.append(save_dict)
            
        save_df = pd.DataFrame(save_dict_list)
        
        return save_df
            
    def _x_csv_path(self, output_dir: str, x_idx: int) -> str:
        return f"{output_dir}/results_{self.config.run_name}_x{x_idx + self.config.x_save_value}.csv"

    def _resume_from_existing(self, output_dir: str) -> None:
        if not self.config.resume:
            return
        last_idx = None
        n_skip = 0
        for x_idx in range(self.num_x_values):
            path = self._x_csv_path(output_dir, x_idx)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                last_idx = x_idx
                n_skip += 1
        if last_idx is None:
            print("Resume: no existing x-slices found, starting from the beginning.")
            return
        last_path = self._x_csv_path(output_dir, last_idx)
        last_df = pd.read_csv(last_path)
        if "api_calls" in last_df.columns and len(last_df):
            self.num_api_calls = int(last_df["api_calls"].iloc[-1])
        print(
            f"Resume: skipping {n_skip}/{self.num_x_values} existing x-slices "
            f"(last kept x{last_idx + self.config.x_save_value}, api_calls={self.num_api_calls})."
        )

    def run_experiment(self):
        output_dir = f"results/toy_regression/{self.config.dataset_name}/{self.config.save_directory}"
        os.makedirs(output_dir, exist_ok=True)
        self._resume_from_existing(output_dir)
        feature_column = self.feature_columns[0]
        for x_idx in range(self.num_x_values):
            csv_path = self._x_csv_path(output_dir, x_idx)
            if self.config.resume and os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
                continue
            save_df = self.process_single_x_value(x_idx)
            save_df.to_csv(csv_path, index=False)
            if self.config.plot:
                x_val = float(self.x_row.iloc[x_idx][feature_column])
                at_checkpoint = abs(x_val / 5.0 - round(x_val / 5.0)) < 1e-6
                at_end = x_idx == self.num_x_values - 1
                if at_checkpoint or at_end:
                    try:
                        plot_paths = save_toy_regression_plots(output_dir, self.config.run_name)
                    except Exception as exc:
                        print(f"Skipping plot refresh at x={x_val}: {exc}")
                    else:
                        for path in plot_paths:
                            print(f"Saved plot at x={x_val}: {path}")
        
def main(argv=None):
    parse_args(argv)
    config = ToyRegressionExperimentConfig(**vars(args))
    experiment = ToyRegressionExperiment(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
