import pandas as pd
import matplotlib.pyplot as plt
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
        print(f"Saving figure to: {output_path} (dpi={dpi}, compression='{compression or 'None'}')")
        figure.savefig(
            output_path,
            format='tiff',
            dpi=dpi,
            bbox_inches='tight', # Apply tight bounding box
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

    # --- Determine Input Path ---
    try:
        script_location = Path(__file__).resolve()
        project_root = script_location.parent.parent
        # Point to the 'data' subfolder within 'results' for the input file
        input_dir = project_root / 'results' / 'data' # <-- CORRECTED path
        input_csv_path = input_dir / "analysis_results.csv"

        print(f"Loading data from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        print("Data loaded successfully.")

    except FileNotFoundError:
        print(f"\nERROR: Input file not found at '{input_csv_path}'")
        print("Verify that the file 'analysis_results.csv' exists in the 'results/data' directory.")
        exit() # Exits the script if the file is not found
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        exit()

    # --- Process Data for Plotting ---
    print("Processing data for bar chart...")
    try:
        # Calculate mean of specified metrics per model
        mean_metrics = df.groupby("model")[["semantic_similarity", "textual_similarity", "contradiction_rate"]].mean().reset_index()

        # Rename columns for plot labels
        mean_metrics = mean_metrics.rename(columns={
            "semantic_similarity": "Semantic Similarity",
            "textual_similarity": "Textual Similarity",
            "contradiction_rate": "Logical Consistency" # Will be inverted next
        })

        # Convert contradiction rate to logical consistency (1 - rate)
        mean_metrics["Logical Consistency"] = 1 - mean_metrics["Logical Consistency"]
        print("Data processed.")

    except KeyError as e:
        print(f"\nERROR: Column '{e}' not found in CSV file '{input_csv_path}'. Check column names.")
        exit()
    except Exception as e:
        print(f"\nERROR processing data for bar chart: {e}")
        exit()

    # --- Generate Bar Chart ---
    print("Generating bar chart...")
    try:
        # Use Pandas plotting (which uses Matplotlib backend)
        ax = mean_metrics.set_index("model").plot(kind="bar", figsize=(10, 6), legend=True)

        # Get the figure object associated with the plot/axes
        fig = plt.gcf() # Get Current Figure

        # Customize plot
        plt.title("Average Self-Reference Consistency per Model")
        plt.ylabel("Score")
        plt.ylim(0, 1) # Set Y-axis limits
        plt.xticks(rotation=0) # Keep model names horizontal
        plt.grid(axis="y", linestyle='--', alpha=0.7) # Add horizontal grid lines
        # plt.tight_layout() # Often handled by bbox_inches='tight' during save

        print("Bar chart generated.")

        # --- Save Figure as TIFF using the helper function ---
        # Filename will be Figure_3.tif, saved in results/
        save_plot_as_tiff(fig, filename="Figure_3.tif", dpi=300, compression='tiff_lzw')

        # plt.show() # Kept commented out, save_plot_as_tiff closes the figure

    except Exception as e:
        print(f"\nERROR during bar chart generation or saving: {e}")
        print("Traceback:")
        print(traceback.format_exc())

    print("\nScript finished.")