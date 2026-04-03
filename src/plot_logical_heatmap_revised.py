import argparse
from pathlib import Path
import traceback

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate the logical consistency heatmap from a prompt-level analysis CSV."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(script_dir / "analysis_results_temp_0_7.csv"),
        help="Path to the prompt-level analysis CSV for the chosen baseline condition.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(script_dir / "figures"),
        help="Directory where the figure will be saved.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="logical_heatmap.pdf",
        help="Output PDF filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    try:
        print(f"Loading data from: {input_csv}")
        df = pd.read_csv(input_csv)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        raise

    try:
        if "logical_consistency" in df.columns:
            value_col = "logical_consistency"
        else:
            if "contradiction_rate" not in df.columns:
                raise KeyError("Neither 'logical_consistency' nor 'contradiction_rate' is present in the CSV.")
            df["logical_consistency"] = 1 - df["contradiction_rate"]
            value_col = "logical_consistency"

        heatmap_data = df.pivot_table(
            index="model",
            columns="category",
            values=value_col,
            aggfunc="mean",
        )

        model_order = ["hermes", "mistral", "stablelm", "openchat", "tinyllama"]
        existing_models = [m for m in model_order if m in heatmap_data.index]
        remaining_models = [m for m in heatmap_data.index if m not in existing_models]
        heatmap_data = heatmap_data.reindex(existing_models + remaining_models)

        category_order = [
            "identity",
            "consciousness",
            "memory",
            "agency",
            "embodiment",
            "morality",
            "introspection",
        ]
        existing_categories = [c for c in category_order if c in heatmap_data.columns]
        remaining_categories = [c for c in heatmap_data.columns if c not in existing_categories]
        heatmap_data = heatmap_data[existing_categories + remaining_categories]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Logical Consistency"},
            ax=ax,
        )
        ax.set_title("Logical Consistency by Model and Category (baseline: temperature = 0.7)")
        ax.set_ylabel("Model")
        ax.set_xlabel("Category")
        fig.tight_layout()

        print(f"Saving figure to: {output_path}")
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"SUCCESS: Figure saved as PDF: {output_path}")
    except Exception as e:
        print(f"ERROR generating or saving heatmap: {e}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()