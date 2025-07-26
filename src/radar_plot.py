import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure # For type hinting
from pathlib import Path
import traceback

# ==============================================================================
#  MAIN SCRIPT BODY
# ==============================================================================

if __name__ == "__main__": # Good practice for organization

    # --- Determine Input Path ---
    try:
        script_location = Path(__file__).resolve()
        project_root = script_location.parent.parent
        # Point to the 'data' subfolder within 'results' for the input file
        input_dir = project_root / 'results' / 'data'
        input_csv_path = input_dir / "analysis_results.csv"

        print(f"Loading data from: {input_csv_path}")
        # Keep the encoding specified in the original script
        df = pd.read_csv(input_csv_path, encoding="utf-8")
        print("Data loaded successfully.")

    except FileNotFoundError:
        print(f"\nERROR: Input file not found at '{input_csv_path}'")
        print("Verify that the file 'analysis_results.csv' exists in the 'results/data' directory.")
        exit() # Exits the script if the file is not found
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        exit()

    # --- Process Data for Radar Plot ---
    print("Processing data for radar plot...")
    try:
        # Group by model and calculate mean metrics
        grouped = df.groupby("model").agg({
            "semantic_similarity": "mean",
            "textual_similarity": "mean",
            "contradiction_rate": "mean"
        }).reset_index()

        # Add logical consistency as 1 - contradiction rate
        grouped["logical_consistency"] = 1 - grouped["contradiction_rate"]

        # Select and define metrics/labels for the radar axes
        metrics = ["semantic_similarity", "textual_similarity", "logical_consistency"]
        labels = ["Semantic Similarity", "Textual Similarity", "Logical Consistency"]
        print("Data processed.")

    except KeyError as e:
        print(f"\nERROR: Column '{e}' not found in CSV file '{input_csv_path}'. Check column names.")
        exit()
    except Exception as e:
        print(f"\nERROR processing data for radar plot: {e}")
        exit()

    # --- Generate Radar Plot ---
    print("Generating radar plot...")
    fig = None # Initialize to handle potential errors before assignment
    try:
        # Calculate angles for the radar axes
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        # Create figure and polar axes
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Plot data for each model
        for i, row in grouped.iterrows():
            values = row[metrics].tolist()
            values += values[:1]  # Close the plot loop
            ax.plot(angles, values, label=row["model"], linewidth=2)
            ax.fill(angles, values, alpha=0.1) # Fill area

        # --- Customize Plot Aesthetics ---
        ax.set_theta_offset(np.pi / 2) # Start axis at the top
        ax.set_theta_direction(-1) # Clockwise direction
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title("Self-Reference Consistency Radar (per Model)", size=14, pad=20)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
        ax.grid(True)
        print("Radar plot generated.")

        # --- Define Output Path e Salva a Figura como PDF ---
        output_dir = project_root / 'results' / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # <<< ALTERAÇÃO AQUI >>>
        output_filename = "radar_plot.pdf"
        output_pdf_path = output_dir / output_filename

        # <<< ALTERAÇÃO AQUI >>>
        print(f"Saving figure as PDF to: {output_pdf_path}")
        plt.savefig(
            output_pdf_path,
            format='pdf', # <<< ALTERAÇÃO AQUI >>>
            bbox_inches='tight'
        )
        # <<< ALTERAÇÃO AQUI >>>
        print(f"SUCCESS: Figure saved as PDF: {output_pdf_path}")

    except Exception as e:
        print(f"\nERROR during radar plot generation or saving: {e}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        if fig:
            plt.close(fig)

    print("\nScript finalizado.")