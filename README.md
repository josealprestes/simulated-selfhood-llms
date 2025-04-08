# LLM Self-Reference Analysis

This repository contains the code, data, and figures for the paper **"Simulated Selfhood in LLMs: A Behavioral Analysis of Introspective Coherence (Preprint Version)"**.

The study evaluates introspective simulation across five open-weight Large Language Models (LLMs), using repeated prompts and a three-stage evaluation pipeline (textual, semantic, and inferential).

## Project Structure

- `src/` — All Python scripts used in the experiment.
- `outputs/` — JSON and CSV files with model responses.
- `results/` — Final figures and computed analysis.
- `models/` — Empty folder with README for model download instructions.
- `requirements.txt` — Python dependencies.
- `LICENSE` — Open license for reuse.

## Reproducing the Experiment

1. Clone this repo:
   ```bash
   git clone https://github.com/SEU_USUARIO/llm-self-reference-analysis.git
   cd llm-self-reference-analysis
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the models (see `models/README.md`).

5. Run the experiment:
   ```bash
   python src/main.py
   ```

## Preprint

This work is publicly available as a preprint on the Open Science Framework (OSF) under DOI [10.31219/osf.io/zhx97_v1](https://doi.org/10.31219/osf.io/zhx97_v1).

For citation and updates, please refer to the OSF version as the primary reference.  
Additional mirrors are also available in other repositories.

## License

Distributed under the MIT License. See `LICENSE` for details.
