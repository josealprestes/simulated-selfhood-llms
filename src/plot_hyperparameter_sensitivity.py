import argparse
from pathlib import Path
import traceback

import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_FILES = {
    0.2: str(SCRIPT_DIR / "analysis_results_temp_0_2_model_summary.csv"),
    0.7: str(SCRIPT_DIR / "analysis_results_temp_0_7_model_summary.csv"),
    1.0: str(SCRIPT_DIR / "analysis_results_temp_1_0_model_summary.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the hyperparameter sensitivity figure comparing temperatures 0.2, 0.7, and 1.0."
    )
    parser.add_argument("--summary_02", type=str, default=DEFAULT_FILES[0.2])
    parser.add_argument("--summary_07", type=str, default=DEFAULT_FILES[0.7])
    parser.add_argument("--summary_10", type=str, default=DEFAULT_FILES[1.0])
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR / "figures"),
        help="Directory where the figure will be saved.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="Figure_3_hyperparameter_sensitivity.pdf",
        help="Output PDF filename.",
    )
    return parser.parse_args()


def load_summary(csv_path: Path, temperature_label: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "logical_consistency" not in df.columns:
        if "contradiction_rate" not in df.columns:
            raise KeyError(f"File {csv_path} lacks both logical_consistency and contradiction_rate.")
        df["logical_consistency"] = 1 - df["contradiction_rate"]
    df["temperature_label"] = temperature_label
    return df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    try:
        data_frames = [
            load_summary(Path(args.summary_02).resolve(), 0.2),
            load_summary(Path(args.summary_07).resolve(), 0.7),
            load_summary(Path(args.summary_10).resolve(), 1.0),
        ]
        df = pd.concat(data_frames, ignore_index=True)
    except Exception as e:
        print(f"ERROR loading summary files: {e}")
        print(traceback.format_exc())
        raise

    try:
        metrics = [
            ("logical_consistency", "Logical Consistency"),
            ("semantic_similarity", "Semantic Similarity"),
            ("diachronic_semantic_similarity", "Diachronic Semantic Similarity"),
        ]
        model_order = ["hermes", "mistral", "stablelm", "openchat", "tinyllama"]
        temperatures = [0.2, 0.7, 1.0]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 12), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, (metric_col, metric_label) in zip(axes, metrics):
            for model in model_order:
                model_df = df[df["model"] == model].sort_values("temperature_label")
                if model_df.empty:
                    continue
                ax.plot(
                    model_df["temperature_label"],
                    model_df[metric_col],
                    marker="o",
                    linewidth=2,
                    label=model,
                )

            ax.set_ylabel(metric_label)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            ax.set_xticks(temperatures)
            ax.set_xticklabels([str(t) for t in temperatures])

        axes[0].set_title("Hyperparameter sensitivity across matched temperature conditions")
        axes[-1].set_xlabel("Temperature")
        axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        fig.tight_layout()
        print(f"Saving figure to: {output_path}")
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"SUCCESS: Figure saved as PDF: {output_path}")
    except Exception as e:
        print(f"ERROR generating hyperparameter sensitivity figure: {e}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()