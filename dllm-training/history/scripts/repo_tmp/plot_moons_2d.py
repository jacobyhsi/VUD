"""Save the eval_toy_2d_class heatmaps for finished moons_1 runs."""
import os
import sys

import colorsys
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))

from notebook_utils import set_plot_style
from src.utils import calculate_min_Va_by_KL_rank

set_plot_style()

ZOOM_LEVEL = 4
NUM_CLASSES = 2
RUNS = [
    "results/toy_classification/moons_1/Dream-v0-Base-7B/va_vanilla",
    "results/toy_classification/moons_1/Dream-v0-Base-7B/va_parallel_pp",
]


def brighter_colour(colour, amount=0.5):
    rgb = mpl.colors.to_rgb(colour)
    hue, light, sat = colorsys.rgb_to_hls(*rgb)
    sat = min(1, sat + amount)
    return mpl.colors.to_hex(colorsys.hls_to_rgb(hue, light, sat))


def load_results(results_directory, num_classes, num_valid_Va=5):
    prob_labels = list(range(num_classes))
    directory = results_directory if results_directory.endswith(os.sep) else results_directory + os.sep

    d_data = None
    frames = []
    for filename in os.listdir(directory):
        if filename.startswith("D_") and filename.endswith(".csv"):
            d_data = pd.read_csv(directory + filename)
        elif filename.startswith("results_") and filename.endswith(".csv"):
            frames.append(pd.read_csv(directory + filename))
    if d_data is None or not frames:
        raise FileNotFoundError(f"Missing D_ or results_ files in {results_directory}")

    for frame in frames:
        calculate_min_Va_by_KL_rank(frame, num_valid_Va=num_valid_Va, forward_kl=True)

    feature_columns = [col for col in d_data.columns if col not in ("label", "note")]
    x_feature_list = []
    total_uncertainty_list = []
    kl_pyx_pyxz_list = []
    kl_pyxz_pyx_list = []
    min_va_list = []
    max_ve_list = []
    z_value_for_min_va_list = []
    prob_y_xz_list = []
    prob_y_x_list = []
    skipped = 0

    for z_df in frames:
        try:
            x_feature_list.append([z_df[f"x_{col}"].values[0] for col in feature_columns])
            total_uncertainty_list.append(z_df["H[p(y|x,D)]"].values[0])
            min_va_index = z_df[z_df["z_value_for_min_Va"]].index[0]
            z_value_for_min_va_list.append(
                [z_df[f"z_{col}"].values[min_va_index] for col in feature_columns]
            )
            prob_y_xz = {}
            prob_y_x = {}
            for label in prob_labels:
                prob_y_xz[label] = z_df[f"p(y={label}|x,z,D)"].values[min_va_index]
                prob_y_x[label] = z_df[f"p(y={label}|x,D)"].values[min_va_index]
            prob_y_xz_list.append(prob_y_xz)
            prob_y_x_list.append(prob_y_x)
            kl_pyx_pyxz_list.append(z_df["kl_pyx_pyxz"].values[min_va_index])
            kl_pyxz_pyx_list.append(z_df["kl_pyxz_pyx"].values[min_va_index])
            min_va_list.append(z_df["min_Va"].values[0])
            max_ve_list.append(z_df["max_Ve"].values[0])
        except Exception:
            skipped += 1

    data = {
        "total_uncertainty": total_uncertainty_list,
        "min_Va": min_va_list,
        "max_Ve": max_ve_list,
        "kl_pyx_pyxz": kl_pyx_pyxz_list,
        "kl_pyxz_pyx": kl_pyxz_pyx_list,
    }
    data.update({
        f"x_{col}": [row[i] for row in x_feature_list]
        for i, col in enumerate(feature_columns)
    })
    data.update({
        f"z_{col}": [row[i] for row in z_value_for_min_va_list]
        for i, col in enumerate(feature_columns)
    })
    for label in prob_labels:
        data[f"p(y={label}|x,z,D)"] = [row[label] for row in prob_y_xz_list]
        data[f"p(y={label}|x,D)"] = [row[label] for row in prob_y_x_list]

    results_df = pd.DataFrame(data).sort_values(by=[f"x_{col}" for col in feature_columns])
    return results_df, d_data, feature_columns, skipped


def figure_shape(feature_columns, results_df):
    feature_lengths = [len(results_df[f"x_{col}"].unique()) for col in feature_columns]
    feature_values = [results_df[f"x_{col}"].unique() for col in feature_columns]
    feature_ranges = [(values[0], values[-1]) for values in feature_values]
    feature_increment = np.round(
        [values[1] - values[0] for values in feature_values], 2
    )
    extents = np.round(
        [feature_ranges[i][0] - feature_increment[i] for i in range(len(feature_columns))]
        + [feature_ranges[i][1] + feature_increment[i] for i in range(len(feature_columns))],
        3,
    )
    extents = extents.reshape((2, len(feature_columns))).T.flatten()
    return feature_lengths, extents


def scatter_context(ax, d_data):
    seen = set()
    colours = {0: "C2", 1: "C3", 2: "C4"}
    for _, row in d_data.iterrows():
        label = int(row["label"])
        legend = None if label in seen else rf"$y={label}$"
        seen.add(label)
        ax.scatter(
            row["x1"], row["x2"],
            c=colours[label], edgecolor="black", s=100, alpha=1, label=legend,
        )
    ax.legend(framealpha=0.75)


def draw_heatmap(values, extents, cmap, d_data, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.matshow(
        zoom(values, ZOOM_LEVEL), origin="lower", extent=extents, cmap=cmap, vmin=0,
    )
    fig.colorbar(image)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("bottom")
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$", rotation=0)
    scatter_context(ax, d_data)
    return fig


def save_run(relative_dir):
    directory = os.path.join(REPO, relative_dir)
    results_df, d_data, feature_columns, skipped = load_results(directory, NUM_CLASSES)
    feature_lengths, extents = figure_shape(feature_columns, results_df)
    expected = int(np.prod(feature_lengths))
    if len(results_df) != expected:
        raise ValueError(
            f"{relative_dir}: {len(results_df)} points, grid {feature_lengths} needs {expected}"
        )

    panels = [
        (
            "plot_total_uncertainty_test.png",
            results_df["total_uncertainty"].values.reshape(feature_lengths).T,
            mpl.colors.LinearSegmentedColormap.from_list(
                "white_to_C0", ["white", brighter_colour("C0", 0.3)]
            ),
            (12, 6),
        ),
        (
            "plot_aleatoric_uncertainty_test.png",
            results_df["min_Va"].values.reshape(feature_lengths).T,
            mpl.colors.LinearSegmentedColormap.from_list(
                "white_to_C1", ["white", brighter_colour("C1", 0.3)]
            ),
            (10, 6),
        ),
        (
            "plot_epistemic_uncertainty_test.png",
            results_df["max_Ve"].values.reshape(feature_lengths).T,
            mpl.colors.LinearSegmentedColormap.from_list(
                "white_to_C5", ["white", brighter_colour("C5", 0.5)]
            ),
            (10, 6),
        ),
        (
            "plot_prediction_test.png",
            results_df["p(y=1|x,D)"].values.reshape(feature_lengths).T,
            mpl.colors.LinearSegmentedColormap.from_list("white_to_C3", ["white", "C3"]),
            (10, 6),
        ),
    ]
    written = []
    for name, values, cmap, figsize in panels:
        fig = draw_heatmap(values, extents, cmap, d_data, figsize)
        path = os.path.join(directory, name)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    print(f"{relative_dir}: {len(results_df)} points, skipped {skipped}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    for run in RUNS:
        save_run(run)
