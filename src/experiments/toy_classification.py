import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass

from src.dataset import load_dataset
from src.bayesian_optimisation import new_candidate
from src.utils import ToyClassificationUtils, calculate_entropy, calculate_kl_divergence, calculate_discrete_variance, calculate_min_Va_by_KL_rank, append_x_results, combined_results_path, load_result_x_frames, resume_completed_x
from src.prompt import ToyClassificationPrompt
from src.chat import chat
from src.parallel.va_estimators import get_va_estimator

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
parser.add_argument("--D_size", default=None, type=int, help="ICL context size. Default 15; moons_1/moons_2 use 30 and spirals uses 200.")

"""X Configuration"""
parser.add_argument("--x_row_method",type=str, default="x_range")
parser.add_argument("--num_x_samples", default=1, type=int)
parser.add_argument("--x_features", default=None)
parser.add_argument("--x_range", default=None, help="Evaluation grid as a dict string of [start, end, step] per feature. Default {'x1': [-12, 12, 0.2]}. Spirals: both features [-4, 4), step 0.1. Moons_1: both features [-1.5, 2.5), step 0.2. Moons_2: x1 [-3, 3.5), x2 [-2.5, 3), step 0.2.")
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
parser.add_argument("--save_directory", default=None, help="Defaults to --model_name.")
parser.add_argument("--x_save_value", default=0, type=int)
parser.add_argument("--num_api_calls_save_value", default=0, type=int)

parser.add_argument("--verbose_output", default=0, type=int)
parser.add_argument("--plot", default=1, type=int, help="Save decomposition plots as results are written, and again at the end. Logistic regression writes the 1D curve. moons_1, moons_2, and spirals write the 2D heatmaps. Default 1.")
parser.add_argument("--resume", default=0, type=int, help="Append each x to one results_{run_name}.csv. With resume 1, skip x values already in that file or in older per-x CSVs.")
parser.add_argument("--va_method", default="vanilla", type=str, choices=("vanilla", "parallel_pf", "parallel_pp", "parallel", "parallel_hybrid"), help="Va estimator: vanilla (forward factorization), parallel_pf (joint-canvas samples scored by the forward conditional), parallel_pp (joint-count entropy), parallel (legacy PF alias), or parallel_hybrid (parallel u with conditional entropy of sequential y|u).")
parser.add_argument("--num_joint_samples", default=None, type=int, help="Joint (u, y) samples for parallel Va. Default 100 for parallel_pp, otherwise 10.")
args = None


DEFAULT_D_SIZE = 15
DEFAULT_X_RANGE = "{'x1': [-12, 12, 0.2]}"
DEFAULT_NUM_JOINT_SAMPLES = 10
PP_NUM_JOINT_SAMPLES = 100
SPIRALS_D_SIZE = 200
SPIRALS_X_RANGE = "{'x1': [-4, 4, 0.1], 'x2': [-4, 4, 0.1]}"
MOONS_D_SIZE = 30
MOONS_X_RANGES = {
    "moons_1": "{'x1': [-1.5, 2.5, 0.2], 'x2': [-1.5, 2.5, 0.2]}",
    "moons_2": "{'x1': [-3.0, 3.5, 0.2], 'x2': [-2.5, 3.0, 0.2]}",
}
TWO_D_PLOT_DATASETS = {"moons_1", "moons_2", "spirals"}


def parse_args(argv=None):
    global args
    args = parser.parse_args(argv)
    if args.dataset_name == "spirals":
        if args.D_size is None:
            args.D_size = SPIRALS_D_SIZE
        if args.x_range is None:
            args.x_range = SPIRALS_X_RANGE
    elif args.dataset_name in MOONS_X_RANGES:
        if args.D_size is None:
            args.D_size = MOONS_D_SIZE
        if args.x_range is None:
            args.x_range = MOONS_X_RANGES[args.dataset_name]
    else:
        if args.D_size is None:
            args.D_size = DEFAULT_D_SIZE
        if args.x_range is None:
            args.x_range = DEFAULT_X_RANGE
    if args.num_joint_samples is None:
        args.num_joint_samples = (
            PP_NUM_JOINT_SAMPLES if args.va_method == "parallel_pp" else DEFAULT_NUM_JOINT_SAMPLES
        )
    save_directory_was_default = args.save_directory is None
    if args.save_directory is None:
        args.save_directory = args.model_name.rsplit("/", 1)[-1]
    if save_directory_was_default:
        args.save_directory = f"{str(args.save_directory).rstrip('/')}/va_{args.va_method}"
    return args

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
    plot: int
    resume: int
    va_method: str
    num_joint_samples: int

def save_toy_classification_plots(output_dir: str, run_name: str) -> list[str]:
    """Save the total-uncertainty plot using the code in eval/eval_toy_1d_class.ipynb."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from csaps import csaps

    eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from notebook_utils import set_plot_style

    set_plot_style()

    results_directory = output_dir if output_dir.endswith(os.sep) else output_dir + os.sep

    D_data = None
    for filename in os.listdir(results_directory):
        if f"D_" in filename and filename.endswith(".csv"):
            D_data = pd.read_csv(results_directory + filename)

    if D_data is None:
        print(f"Skipping plot: missing ICL file in {output_dir}")
        return []

    feature_column = [col for col in D_data.columns if col != "label" and col != "note"][0]

    df_list = load_result_x_frames(output_dir)

    if not df_list:
        print(f"Skipping plot: no results_*.csv files in {output_dir}")
        return []

    for save_data in df_list:
        if "Var[p(y|x,D)]" in save_data.columns:
            save_data["Var[y|x,D]"] = save_data["Var[p(y|x,D)]"]
        save_data = calculate_min_Va_by_KL_rank(save_data, num_valid_Va=5, forward_kl=True, upper_bound_by_total_U=True)
        save_data = calculate_min_Va_by_KL_rank(save_data, num_valid_Va=5, forward_kl=True, upper_bound_by_total_U=True, uncertainty_type="variance")

    x_x1_list = []
    total_uncertainty_list = []
    min_Va_list = []
    max_Ve_list = []
    kl_pyx_pyxz_list = []
    within_threshold_list = []
    total_variance_list = []
    min_Va_variance_list = []
    max_Ve_variance_list = []

    for z_df in df_list:
        try:
            x_x1 = z_df[f"x_{feature_column}"].values[0]
            x_x1_list.append(x_x1)
            total_uncertainty = z_df["H[p(y|x,D)]"].values[0]
            total_uncertainty_list.append(total_uncertainty)

            min_Va = z_df["min_Va"].values[0]
            min_Va_list.append(min_Va)
            max_Ve = z_df["max_Ve"].values[0]
            max_Ve_list.append(max_Ve)

            within_threshold_list.append(z_df[z_df["within_threshold"]][f"z_{feature_column}"].values)

            min_Va_index = z_df[z_df["z_value_for_min_Va"]].index[0]

            kl_pyx_pyxz = z_df["kl_pyx_pyxz"].values[min_Va_index]
            kl_pyx_pyxz_list.append(kl_pyx_pyxz)

        except Exception:
            pass

        try:
            total_variance = z_df["Var[y|x,D]"].values[0]
            total_variance_list.append(total_variance)

            min_Va_variance = z_df["min_Va_variance"].values[0]
            min_Va_variance_list.append(min_Va_variance)

            max_Ve_variance = z_df["max_Ve_variance"].values[0]
            max_Ve_variance_list.append(max_Ve_variance)
        except Exception:
            pass

    data = {
        f"x_{feature_column}": x_x1_list,
        "total_uncertainty": total_uncertainty_list,
        "min_Va": min_Va_list,
        "max_Ve": max_Ve_list,
        "within_threshold": within_threshold_list,
        "kl_pyx_pyxz": kl_pyx_pyxz_list,
    }

    if len(total_variance_list) > 0:
        data["total_variance"] = total_variance_list
        data["min_Va_variance"] = min_Va_variance_list
        data["max_Ve_variance"] = max_Ve_variance_list

    num_Va = len(min_Va_list)
    for key in data.keys():
        data[key] = data[key][:num_Va]

    results_df = pd.DataFrame(data)
    results_df = results_df.sort_values(by=f"x_{feature_column}")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    plt.scatter(results_df[f"x_{feature_column}"], results_df["total_uncertainty"], color="C0", alpha=0.4)
    plt.scatter(results_df[f"x_{feature_column}"], results_df["min_Va"], color="C1", alpha=0.4)

    # line of best fit

    x_grid = np.linspace(results_df[f"x_{feature_column}"].min(), results_df[f"x_{feature_column}"].max(), 100)
    y_total_uncertainty = csaps(results_df[f"x_{feature_column}"], results_df["total_uncertainty"], smooth=0.85)
    y_min_Va = csaps(results_df[f"x_{feature_column}"], results_df["min_Va"], smooth=0.85)

    plt.plot(x_grid, y_total_uncertainty(x_grid), color="C0", linewidth=3, label="Total Uncertainty")
    plt.plot(x_grid, y_min_Va(x_grid), color="C1", linewidth=3, label="Aleatoric Uncertainty")

    # vertical line for the true x
    label_0_seen = False
    label_1_seen = False
    for i, row in D_data.iterrows():
        if row["label"] == 0:
            if not label_0_seen:
                label_0_seen = True
                label_string = r"ICL Data: $y = 0$"
            else:
                label_string = None
            plt.axvline(x=row[feature_column], color="C2", linestyle="--", linewidth=3, alpha=0.3, label=label_string)
        else:
            if not label_1_seen:
                label_1_seen = True
                label_string = r"ICL Data: $y = 1$"
            else:
                label_string = None
            plt.axvline(x=row[feature_column], color="C3", linestyle="--", linewidth=3, alpha=0.3, label=label_string)

    plt.title("Total Uncertainty Decomposition: Toy Classification")
    plt.ylabel("Uncertainty")
    plt.xlabel(r"Test Covariate $x$")

    plt.legend(framealpha=0.75)

    plot_path = os.path.join(output_dir, f"plot_entropy_decomposition_{run_name}.png")
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [plot_path]


def save_2d_classification_plots(output_dir: str, run_name: str) -> list[str]:
    """Save 2D classification heatmaps from the results CSV, including a partial grid."""
    import colorsys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from notebook_utils import set_plot_style

    set_plot_style()

    results_directory = output_dir if output_dir.endswith(os.sep) else output_dir + os.sep
    d_data = None
    for filename in os.listdir(results_directory):
        if filename.startswith("D_") and filename.endswith(".csv"):
            d_data = pd.read_csv(results_directory + filename)
    frames = load_result_x_frames(output_dir)
    if d_data is None:
        print(f"Skipping plot: missing ICL file in {output_dir}")
        return []
    if not frames:
        print(f"Skipping plot: no results_*.csv files in {output_dir}")
        return []

    for frame in frames:
        calculate_min_Va_by_KL_rank(frame, num_valid_Va=5, forward_kl=True)

    feature_columns = [col for col in d_data.columns if col not in ("label", "note")]
    if len(feature_columns) != 2:
        print(f"Skipping 2D plot: expected 2 features, found {feature_columns}")
        return []

    prob_labels = [
        column[len("p(y="):-len("|x,D)")]
        for column in frames[0].columns
        if column.startswith("p(y=") and column.endswith("|x,D)")
    ]
    rows = []
    skipped = 0
    for z_df in frames:
        try:
            min_va_index = z_df[z_df["z_value_for_min_Va"]].index[0]
            row = {
                "x1": z_df[f"x_{feature_columns[0]}"].values[0],
                "x2": z_df[f"x_{feature_columns[1]}"].values[0],
                "total_uncertainty": z_df["H[p(y|x,D)]"].values[0],
                "min_Va": z_df["min_Va"].values[0],
                "max_Ve": z_df["max_Ve"].values[0],
            }
            for label in prob_labels:
                row[f"p_y{label}"] = z_df[f"p(y={label}|x,D)"].values[min_va_index]
            rows.append(row)
        except Exception:
            skipped += 1
    if not rows:
        print(f"Skipping plot: no usable 2D results in {output_dir}")
        return []

    points = pd.DataFrame(rows)
    counts = points.groupby("x1").size()
    complete_x1 = counts[counts == counts.max()].index
    points = points[points["x1"].isin(complete_x1)]
    x1_values = np.sort(points["x1"].unique())
    x2_values = np.sort(points["x2"].unique())
    grid = points.pivot(index="x2", columns="x1", values="total_uncertainty")
    if grid.shape != (len(x2_values), len(x1_values)) or grid.isna().any().any():
        print(f"Skipping plot: 2D results are not a complete grid ({len(points)} points).")
        return []

    def panel_values(column):
        return points.pivot(index="x2", columns="x1", values=column).to_numpy()

    step = np.round(np.array([
        x1_values[1] - x1_values[0] if len(x1_values) > 1 else 0.2,
        x2_values[1] - x2_values[0] if len(x2_values) > 1 else 0.2,
    ]), 2)
    extents = np.round([
        x1_values[0] - step[0], x1_values[-1] + step[0],
        x2_values[0] - step[1], x2_values[-1] + step[1],
    ], 3).tolist()

    def brighter_colour(colour, amount):
        hue, light, sat = colorsys.rgb_to_hls(*mpl.colors.to_rgb(colour))
        return mpl.colors.to_hex(colorsys.hls_to_rgb(hue, light, min(1, sat + amount)))

    def scatter_context(ax):
        seen = set()
        colours = {0: "C2", 1: "C3", 2: "C4"}
        for _, row in d_data.iterrows():
            label = int(row["label"])
            legend = None if label in seen else rf"$y={label}$"
            seen.add(label)
            ax.scatter(
                row[feature_columns[0]], row[feature_columns[1]],
                c=colours[label], edgecolor="black", s=100, alpha=1, label=legend,
            )
        ax.legend(framealpha=0.75)

    panels = [
        (
            f"plot_total_uncertainty_{run_name}.png",
            panel_values("total_uncertainty"),
            mpl.colors.LinearSegmentedColormap.from_list(
                "white_to_C0", ["white", brighter_colour("C0", 0.3)]
            ),
            (12, 6),
        ),
        (
            f"plot_aleatoric_uncertainty_{run_name}.png",
            panel_values("min_Va"),
            mpl.colors.LinearSegmentedColormap.from_list(
                "white_to_C1", ["white", brighter_colour("C1", 0.3)]
            ),
            (10, 6),
        ),
        (
            f"plot_epistemic_uncertainty_{run_name}.png",
            panel_values("max_Ve"),
            mpl.colors.LinearSegmentedColormap.from_list(
                "white_to_C5", ["white", brighter_colour("C5", 0.5)]
            ),
            (10, 6),
        ),
    ]
    written = []
    for name, values, cmap, figsize in panels:
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.matshow(zoom(values, 4), origin="lower", extent=extents, cmap=cmap, vmin=0)
        fig.colorbar(image)
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("bottom")
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$", rotation=0)
        scatter_context(ax)
        ax.set_xlim(extents[0], extents[1])
        ax.set_ylim(extents[2], extents[3])
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    if skipped:
        print(f"2D plot skipped {skipped} result files in {output_dir}")
    return written


class ToyClassificationExperiment:
    def __init__(self, config: ToyClassificationExperimentConfig):
        self.config = config
        
        np.random.seed(self.config.numpy_seed)

        self.prompter = ToyClassificationPrompt(model_name=self.config.model_name)
        
        if self.config.num_bo_z > self.config.num_z:
            raise ValueError("Number of bo z values cannot be greater than number of z values.")
        if self.config.num_bo_z < 0:
            raise ValueError("Number of bo z values cannot be negative.")

        self.data_preprocessing()
        
        self.num_api_calls = self.config.num_api_calls_save_value
        self.va_estimator = get_va_estimator(self.config.va_method, self.config.num_joint_samples)
        print(f"Va estimator: {self.va_estimator.name}")

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

        output_dir = f"results/toy_classification/{self.config.dataset_name}/{self.config.save_directory}"
        os.makedirs(output_dir, exist_ok=True)
        D_rows.to_csv(f"{output_dir}/D_{self.config.run_name}.csv", index=False)
    
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
            except Exception as exc:
                print(
                    f"Seed {seed + 1} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            self.num_api_calls += 1

        if successful_seeds == 0:
            raise ValueError(
                f"All seeds failed for {probability_calculated}. "
                "Check the model server and request format."
            )

        avg_probs = {label: prob / successful_seeds for label, prob in avg_probs.items()}
        
        if self.config.verbose_output:
            print(f"\nAveraged {probability_calculated} probabilities: {avg_probs}")
            
        return avg_probs

    def score_query_probs(
        self,
        query_note: str,
        permutation_seed: int,
        icl_z_note: Optional[str] = None,
        icl_u_label: Optional[str | int] = None,
    ) -> dict:
        """One-shot p(y|x,u,z,D) score used after a joint u sample."""
        prompt = self.prompter.get_general_prompt(
            D_df=self.D_note_label_df,
            query_note=query_note,
            permutation_seed=permutation_seed,
            icl_z_note=icl_z_note,
            icl_u_label=icl_u_label,
        )
        _pred, probs = chat(
            prompt,
            self.label_keys,
            seed=permutation_seed,
            model=self.config.model_name,
            port=self.config.model_port,
            ip=self.config.model_ip,
            temperature=self.config.model_temperature,
            is_local_client=self.config.is_local_client,
        )
        self.num_api_calls += 1
        return probs
    
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

            z_fields = self.va_estimator.estimate(
                self, x, z, avg_pyx_probs, Hyx, total_variance
            )
            self.z_BO_maximisation_objective.append(-z_fields["Va"] - z_fields["kl_pyx_pyxz"])

            save_dict = {f"z_{feature}": row[feature] for feature in self.feature_columns}
            save_dict["z_note"] = z
            save_dict_x = {f"x_{feature}": self.x_row.iloc[x_idx][feature] for feature in self.feature_columns}
            save_dict_x["x_note"] = x
            save_dict = {**save_dict, **save_dict_x}
            for label, prob in avg_pyx_probs.items():
                save_dict[f"p(y={label}|x,D)"] = prob
            save_dict["H[p(y|x,D)]"] = Hyx
            save_dict["Var[y|x,D]"] = total_variance
            save_dict.update(z_fields)
            save_dict["api_calls"] = self.num_api_calls

            save_dict_list.append(save_dict)
            
        save_df = pd.DataFrame(save_dict_list)
        
        return save_df
            
    def _resume_from_existing(self, output_dir: str) -> None:
        results_path = combined_results_path(output_dir, self.config.run_name)
        if not self.config.resume:
            if results_path.is_file():
                results_path.unlink()
            self._completed_x = set()
            print(f"Writing results to {results_path}")
            return
        self._completed_x, api_calls, dropped = resume_completed_x(
            output_dir,
            self.config.run_name,
            self.x_row["note"].tolist(),
            self.config.num_z,
            self.config.x_save_value,
        )
        if api_calls is not None:
            self.num_api_calls = api_calls
        if not self._completed_x:
            print("Resume: no existing x-slices found, starting from the beginning.")
        else:
            print(
                f"Resume: skipping {len(self._completed_x)}/{self.num_x_values} existing x-slices "
                f"(api_calls={self.num_api_calls})."
            )
        if dropped:
            print(f"Resume: dropped {dropped} rows from a partial trailing x in {results_path.name}.")
        print(f"Writing results to {results_path}")

    def run_experiment_default(self):
        output_dir = f"results/toy_classification/{self.config.dataset_name}/{self.config.save_directory}"
        os.makedirs(output_dir, exist_ok=True)
        self._resume_from_existing(output_dir)
        feature_column = self.feature_columns[0]
        for x_idx in range(self.num_x_values):
            if x_idx in self._completed_x:
                continue
            save_df = self.process_single_x_value(x_idx)
            append_x_results(output_dir, self.config.run_name, save_df)
            if self.config.plot:
                x_val = float(self.x_row.iloc[x_idx][feature_column])
                at_checkpoint = abs(x_val / 5.0 - round(x_val / 5.0)) < 1e-6
                at_end = x_idx == self.num_x_values - 1
                if self.config.dataset_name in TWO_D_PLOT_DATASETS and len(self.feature_columns) == 2:
                    n_x2 = int(self.x_row[self.feature_columns[1]].nunique())
                    at_checkpoint = at_checkpoint or (n_x2 > 0 and x_idx % n_x2 == n_x2 - 1)
                if at_checkpoint or at_end:
                    try:
                        plot_paths = self._save_plots(output_dir)
                    except Exception as exc:
                        print(f"Skipping plot refresh at x={x_val}: {exc}")
                    else:
                        for path in plot_paths:
                            print(f"Saved plot at x={x_val}: {path}")
    
    def run_experiment(self):
        self.run_experiment_default()
        
        print(f"Total API Calls: {self.num_api_calls}")
        
        output_dir = f"results/toy_classification/{self.config.dataset_name}/{self.config.save_directory}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/api_calls_{self.config.run_name}.txt", "w") as f:
            f.write(f"Total API Calls: {self.num_api_calls}")

        if self.config.plot:
            output_dir = f"results/toy_classification/{self.config.dataset_name}/{self.config.save_directory}"
            try:
                plot_paths = self._save_plots(output_dir)
            except ImportError as exc:
                print(f"Skipping plot: {exc}. Install matplotlib and csaps to enable --plot.")
            except Exception as exc:
                print(f"Skipping plot: {exc}")
            else:
                for path in plot_paths:
                    print(f"Saved plot: {path}")

    def _save_plots(self, output_dir: str) -> list[str]:
        if self.config.dataset_name in TWO_D_PLOT_DATASETS:
            return save_2d_classification_plots(output_dir, self.config.run_name)
        return save_toy_classification_plots(output_dir, self.config.run_name)

def main(argv=None):
    parse_args(argv)
    config = ToyClassificationExperimentConfig(**vars(args))
    experiment = ToyClassificationExperiment(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
