import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        # Input still comes from 'results/data/'
        input_dir = project_root / 'results' / 'data'
        input_csv_path = input_dir / "analysis_results.csv"

        print(f"Loading data from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        print("Data loaded successfully.")

    except FileNotFoundError:
        print(f"\nERROR: Input file not found at '{input_csv_path}'")
        print("Verify that the file 'analysis_results.csv' exists in the 'results/data' directory.")
        exit()
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        exit()

    # --- Prepare Data for Heatmap ---
    print("Preparing data for the heatmap...")
    try:
        heatmap_data = df.pivot_table(index="model", columns="category", values="contradiction_rate", aggfunc="mean")
        heatmap_data = 1 - heatmap_data  # convert to consistency
        print("Heatmap data prepared.")
    except KeyError as e:
        print(f"\nERROR: Column '{e}' not found in CSV file '{input_csv_path}'. Check column names.")
        exit()
    except Exception as e:
        print(f"\nERROR processing data for the heatmap: {e}")
        exit()

    # --- Generate Heatmap ---
    print("Generating heatmap...")
    fig = None # Initialize fig to None for finally block
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1,
                    cbar_kws={"label": "Logical Consistency"}, ax=ax)

        ax.set_title("Logical Consistency by Model and Category")
        ax.set_ylabel("Model")
        ax.set_xlabel("Category")
        fig.tight_layout() # Apply tight layout

        print("Heatmap generated.")

        # --- Define o caminho de saída e salva como PDF ---
        output_dir = project_root / 'results' / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True) 
        
        # <<< ALTERAÇÃO AQUI >>>
        output_filename = "logical_heatmap.pdf" 
        output_pdf_path = output_dir / output_filename 

        # --- Salva a figura diretamente como PDF ---
        # <<< ALTERAÇÃO AQUI >>>
        print(f"Saving figure as PDF to: {output_pdf_path}")
        plt.savefig(
            output_pdf_path,
            format='pdf',           # <<< ALTERAÇÃO AQUI >>>
            bbox_inches='tight'
        )
        # <<< ALTERAÇÃO AQUI >>>
        print(f"SUCCESS: Figure saved as PDF: {output_pdf_path}")

    except Exception as e:
        print(f"\nERROR during heatmap generation or saving: {e}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        # --- Close the figure to free memory ---
        if fig is not None:
            plt.close(fig)

    print("\nScript finalizado.")