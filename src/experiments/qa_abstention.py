import argparse
import math
from pathlib import Path

import pandas as pd


UNCERTAINTY_COLUMNS = ("TU", "Va", "Ve")


def retained_percentage(value: str) -> float:
    percentage = float(value)
    if not 0 < percentage <= 100:
        raise argparse.ArgumentTypeError("retained percentages must be in (0, 100]")
    return percentage


def extract_id_results(results: pd.DataFrame) -> pd.DataFrame:
    if "is_ood" not in results.columns:
        raise ValueError("Cannot compute abstention; missing column: is_ood")

    is_ood = pd.to_numeric(results["is_ood"], errors="coerce")
    invalid_is_ood = is_ood.isna() | ~is_ood.isin([0, 1])
    if invalid_is_ood.any():
        raise ValueError("The is_ood column must contain only 0 or 1 values.")

    id_results = results.loc[is_ood.eq(0)].copy()
    if id_results.empty:
        raise ValueError("The input file does not contain any ID rows (is_ood = 0).")

    id_results["is_ood"] = 0
    return id_results


def compute_abstention_results(
    results: pd.DataFrame,
    retained_percentages,
) -> pd.DataFrame:
    required_columns = {
        "is_ood",
        "true_label",
        "pred_label",
        *UNCERTAINTY_COLUMNS,
    }
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot compute abstention; missing columns: {sorted(missing_columns)}"
        )
    if results.empty:
        raise ValueError("Cannot compute abstention on an empty result set.")
    if not results["is_ood"].eq(0).all():
        raise ValueError("Abstention results must contain only is_ood = 0 rows.")

    evaluated = results.copy()
    for uncertainty in UNCERTAINTY_COLUMNS:
        evaluated[uncertainty] = pd.to_numeric(
            evaluated[uncertainty],
            errors="coerce",
        )
    if evaluated[list(UNCERTAINTY_COLUMNS)].isna().any().any():
        raise ValueError("Cannot compute abstention with missing uncertainty values.")

    true_labels = evaluated["true_label"].map(lambda value: str(value).strip())
    pred_labels = evaluated["pred_label"].map(lambda value: str(value).strip())
    evaluated["correct"] = true_labels.eq(pred_labels)
    baseline_accuracy = float(evaluated["correct"].mean())

    rows = []
    total_count = len(evaluated)
    for uncertainty in UNCERTAINTY_COLUMNS:
        ranked = evaluated.sort_values(
            uncertainty,
            ascending=True,
            kind="mergesort",
        )
        for percentage in retained_percentages:
            retained_count = max(
                1,
                math.ceil(total_count * float(percentage) / 100.0),
            )
            retained = ranked.head(retained_count)
            filtered_accuracy = float(retained["correct"].mean())
            improvement = filtered_accuracy - baseline_accuracy
            rows.append(
                {
                    "uncertainty": uncertainty,
                    "retained_percentage": float(percentage),
                    "retained_count": retained_count,
                    "removed_count": total_count - retained_count,
                    "baseline_accuracy": baseline_accuracy,
                    "filtered_accuracy": filtered_accuracy,
                    "accuracy_improvement": improvement,
                    "accuracy_improvement_pp": improvement * 100.0,
                }
            )

    return pd.DataFrame(rows)


def print_abstention_report(report: pd.DataFrame) -> None:
    baseline_accuracy = report["baseline_accuracy"].iloc[0]
    total_count = int(
        report["retained_count"].max() + report["removed_count"].min()
    )
    print(f"\nBaseline accuracy ({total_count} questions): {baseline_accuracy:.4f}")
    for uncertainty in UNCERTAINTY_COLUMNS:
        subset = report[report["uncertainty"] == uncertainty].copy()
        subset["accuracy"] = subset["filtered_accuracy"].map(
            lambda value: f"{value:.4f}"
        )
        subset["improvement_pp"] = subset["accuracy_improvement_pp"].map(
            lambda value: f"{value:+.2f}"
        )
        print(f"\nFiltering by {uncertainty} (highest uncertainty removed):")
        print(
            subset[
                [
                    "retained_percentage",
                    "retained_count",
                    "removed_count",
                    "accuracy",
                    "improvement_pp",
                ]
            ].to_string(index=False)
        )


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.startswith("df_"):
        stem = stem[len("df_"):]
    return input_path.with_name(f"abstention_{stem}.csv")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output is not None
        else default_output_path(input_path)
    )

    if input_path.resolve() == output_path.resolve():
        raise ValueError("The output path must be different from the input path.")

    results = pd.read_csv(input_path)
    id_results = extract_id_results(results)
    report = compute_abstention_results(
        id_results,
        args.retained_percentages,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print(f"Loaded {len(id_results)} ID rows from {len(results)} total rows.")
    print_abstention_report(report)
    print(f"\nSource results: {input_path}")
    print(f"Abstention results: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute QA abstention metrics from existing run.py qa results."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an ID-only or mixed ID/OOD result CSV from run.py qa.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output CSV path. By default, an abstention_*.csv file is written "
            "beside the input file."
        ),
    )
    parser.add_argument(
        "--retained-percentages",
        nargs="+",
        type=retained_percentage,
        default=list(range(100, 0, -10)),
        metavar="PERCENT",
        help=(
            "Percentages of lowest-uncertainty ID questions to retain "
            "(default: 100 90 ... 10)."
        ),
    )
    return parser


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    main()
