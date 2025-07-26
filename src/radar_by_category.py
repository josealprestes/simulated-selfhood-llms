import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json # Added for JSON loading
import matplotlib.figure # For type hinting
from pathlib import Path
import traceback

# ==============================================================================
#  HELPER FUNCTION TO SAVE PLOTS AS TIFF (Same as before)
# ==============================================================================

def save_plot_as_tiff(figure: matplotlib.figure.Figure, filename: str, dpi: int = 300, compression: str = 'tiff_lzw'):
    """
    Saves a Matplotlib figure as a TIFF file in the project's 'results' directory.
    Assumes this script is running from within the 'src' folder.
    Args:
        figure: The Matplotlib figure object.
        filename: Output file name (e.g., 'plot1.tif'). '.tif' will be appended if necessary.
        dpi: Image resolution in dots per inch.
        compression: TIFF compression method ('tiff_lzw', 'tiff_deflate', 'None', etc.).
    """
    try:
        # --- Validate filename ---
        if not (filename.lower().endswith('.tif') or filename.lower().endswith('.tiff')):
            original_filename = filename
            filename += '.tif'
            print(f"Warning: '.tif' extension automatically added to filename '{original_filename}'. New name: '{filename}'")

        # --- Determine paths (output goes to 'results/') ---
        script_location = Path(__file__).resolve()
        project_root = script_location.parent.parent
        results_dir = project_root / 'results'
        results_dir.mkdir(parents=True, exist_ok=True) # Create results folder if it doesn't exist
        output_path = results_dir / filename # Full path for the output TIFF

        # --- Save the figure ---
        # Note: Applying tight_layout() before saving is often good for complex plots
        figure.tight_layout(rect=[0, 0.03, 1, 0.97]) # Apply tight layout slightly adjusted for suptitle
        print(f"Saving figure to: {output_path} (dpi={dpi}, compression='{compression or 'None'}')")
        figure.savefig(
            output_path,
            format='tiff',
            dpi=dpi,
            # bbox_inches='tight' might conflict with fig.tight_layout(), using tight_layout above
            pil_kwargs={'compression': compression} if compression and compression.lower() != 'none' else {} # Pass compression options
        )
        print(f"SUCCESS: Figure saved as TIFF: {output_path}")

    except ImportError:
        print("\nERROR: Pillow library not found. Required for saving TIFF with options.")
        print("Install with: pip install Pillow\n")
    except Exception as e:
        print(f"\nERROR saving figure '{filename}' as TIFF: {e}")
        print("Traceback:")
        print(traceback.format_exc()) # Print error details
    finally:
        # Close the figure to free memory after saving (important for scripts)
        plt.close(figure)

# ==============================================================================
#  MAIN SCRIPT BODY
# ==============================================================================

if __name__ == "__main__": # Good practice for organization

    # --- Determine Input Path and Load JSON Data ---
    try:
        script_location = Path(__file__).resolve()
        project_root = script_location.parent.parent
        # Point to the 'data' subfolder within 'results' for the input JSON file
        input_dir = project_root / 'results' / 'data' # <-- CORRECTED path
        input_json_path = input_dir / "analysis_results.json" # <-- Input is JSON

        print(f"Loading data from: {input_json_path}")
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f) # Load data using json module
        print("Data loaded successfully.")

        # Convert loaded JSON data to DataFrame
        df = pd.DataFrame(data)
        print("Data converted to DataFrame.")

    except FileNotFoundError:
        print(f"\nERROR: Input file not found at '{input_json_path}'")
        print("Verify that the file 'analysis_results.json' exists in the 'results/data' directory.")
        exit() # Exits the script if the file is not found
    except json.JSONDecodeError:
        print(f"\nERROR: Could not decode JSON from file '{input_json_path}'. Check file content.")
        exit()
    except Exception as e:
        print(f"\nERROR loading or parsing data: {e}")
        exit()

    # --- Process Data for Plotting ---
    print("Processing data for radar plots...")
    try:
        # Calculate logical consistency
        df["logical_consistency"] = 1 - df["contradiction_rate"]

        # Group by model and category, calculate means
        grouped = df.groupby(["model", "category"]).agg({
            "semantic_similarity": "mean",
            "textual_similarity": "mean",
            "logical_consistency": "mean"
        }).reset_index()

        # --- Define Plotting Parameters ---
        categories = sorted(df["category"].unique())
        metrics = ["semantic_similarity", "textual_similarity", "logical_consistency"]
        metric_labels = ["Semantic", "Textual", "Logical"] # Shorter labels for axes
        models = grouped["model"].unique()

        # Calculate angles for radar axes
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1] # Close the circle
        print("Data processed and parameters defined.")

    except KeyError as e:
        print(f"\nERROR: Column '{e}' not found in the data. Check JSON structure or column names.")
        exit()
    except Exception as e:
        print(f"\nERROR processing data: {e}")
        exit()

    # --- Generate Multiple Radar Plots ---
    print(f"Generating {len(models)} radar plots...")
    try:
        # Create subplots (adjust rows/cols if needed)
        # Assuming max 6 models for a 2x3 grid
        n_rows = 2
        n_cols = 3
        if len(models) > n_rows * n_cols:
             print(f"Warning: More models ({len(models)}) than available subplots ({n_rows * n_cols}). Some models might not be plotted.")
             # Or adjust n_rows/n_cols dynamically

        fig, axs = plt.subplots(n_rows, n_cols, subplot_kw=dict(polar=True), figsize=(18, 10))
        axs = axs.flatten() # Flatten the 2D array of axes for easy iteration

        # Plot data for each model
        for idx, model in enumerate(models):
            if idx >= len(axs): # Stop if we run out of subplots
                 break
            ax = axs[idx]
            model_data = grouped[grouped["model"] == model]

            # Inner loop: Plot each category within the model's subplot
            for cat in categories:
                values = model_data[model_data["category"] == cat][metrics].values
                if values.size == 0: # Skip if no data for this category/model
                    continue
                stats = values.flatten().tolist()
                stats += stats[:1] # Close the plot loop
                # Use consistent colors per category if possible (needs defining a color map)
                ax.plot(angles, stats, label=cat, linewidth=1.5) # Plot category line
                ax.fill(angles, stats, alpha=0.1) # Fill area

            # --- Customize Subplot Aesthetics ---
            ax.set_title(f"{model}\nConsistency by Category", size=10, pad=15) # Model title
            ax.set_xticks(angles[:-1]) # Set tick positions
            ax.set_xticklabels(metric_labels) # Set tick labels
            ax.set_yticks(np.arange(0.2, 1.1, 0.2)) # Y-axis ticks (0.2 to 1.0)
            ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.2, 1.1, 0.2)]) # Y-axis labels
            ax.set_ylim(0, 1) # Y-axis limits

        # Hide unused subplots if any
        for i in range(len(models), len(axs)):
            axs[i].set_visible(False)

        # --- Add Shared Legend and Main Title ---
        # Use handles and labels from one of the subplots for the legend
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(categories)) # Shared legend at bottom

        fig.suptitle("Self-Reference Consistency per Category (Radar per Model)", fontsize=16)

        # Apply tight layout *before* saving
        # fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to prevent overlap with suptitle/legend

        print(f"Generated {min(len(models), len(axs))} radar plots.")

        # --- Save Figure as TIFF using the helper function ---
        # Filename will be Figure_2.tif, saved in results/
        save_plot_as_tiff(fig, filename="Figure_2.tif", dpi=300, compression='tiff_lzw')

        # plt.show() # Kept commented out, save_plot_as_tiff closes the figure

    except Exception as e:
        print(f"\nERROR during radar plot generation or saving: {e}")
        print("Traceback:")
        print(traceback.format_exc())

    print("\nScript finished.")