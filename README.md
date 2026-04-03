# Simulated Selfhood in LLMs: A Behavioral Analysis of Introspective Coherence

Code, data, and analysis package for the study on self-referential consistency in stateless large language models (LLMs).

## Overview

This repository accompanies the study of introspection-like and self-referential outputs in stateless LLMs from a strictly behavioral perspective. The project evaluates whether repeated self-focused prompts elicit stable linguistic regularities or whether the observed patterns are better interpreted as fragile, regime-sensitive generative effects.

**The repository contains:**

* **Source code** for generation, analysis, and figure production.
* **Prompt sets** used in the experiments.
* **Aggregated analysis outputs** for the matched temperature conditions.
* **Final figures** used in the manuscript.
* **Anonymized human-evaluation package** for construct validation.

> [!IMPORTANT]
> The repository does **not** include model weight files.

---

## Study Design Summary

* **Models:** 5 open-weight stateless LLMs
* **Prompts:** 21 introspective prompts
* **Repetitions:** 10 repetitions per prompt
* **Temperature Conditions:**
    * `temperature = 0.2`
    * `temperature = 0.7`
    * `temperature = 1.0`
* **Fixed Decoding Parameters:**
    * `top_p = 0.95`
    * `max_tokens = 100`

**Total yield:** 3,150 completions across all models and matched temperature conditions.

---

## Repository Structure

```text
.
├── analysis/
│   ├── figures/
│   └── results/
├── data/
│   └── human_evaluation/
├── models/
├── outputs/
│   ├── temp_0_2/
│   ├── temp_0_7/
│   └── temp_1_0/
├── src/
├── LICENSE
├── LICENSE-data.md
├── CITATION.cff
├── README.md
└── requirements.txt

``````

## What is Included

### Source Code
The `src/` directory contains scripts for:
* Prompt generation
* Local model execution
* Result aggregation
* Human-vs-automated validation support
* Figure generation

### Analysis Results
The `analysis/results/` directory contains aggregated CSV/JSON outputs for the matched temperature conditions and model summaries.

### Figures
The `analysis/figures/` directory contains the final figure files used in the manuscript, including:
* Matched temperature comparison
* Semantic heatmap
* Logical consistency heatmap

### Human Evaluation Package
The `data/human_evaluation/` directory contains the anonymized materials used for the human-validation layer, including:
* `stimuli_pairs_en_us.csv`
* `human_judgments_long_en_us.csv`
* `rater_metadata_anonymized_en_us.csv`
* `annotation_guidelines_en_us.pdf`

*Optional qualitative feedback may be provided in summarized rather than verbatim form to preserve anonymity.*

---

## What is Not Included
This repository does **not** bundle the model checkpoints because of file size and upstream licensing constraints.

To reproduce the local generations, download the exact **GGUF checkpoints** listed in `models/README.md` and place them in the `models/` directory.

---

## Environment
**Recommended environment:**
* Python 3.11
* Local execution environment with sufficient RAM for the selected GGUF models.

**Install dependencies with:**
```bash
pip install -r requirements.txt

``````

## Reproducibility Workflow

A typical workflow is:
1. Download the required model files into `models/`.
2. Run the generation scripts in `src/`.
3. Run the analysis scripts.
4. Regenerate figures from the aggregated outputs.

> **Note:** If you are primarily interested in verifying the published analyses rather than regenerating all model outputs, the contents of `analysis/results/` and `analysis/figures/` should be sufficient.

---

## Human Evaluation Notes

The human-evaluation layer is included as an anonymized construct-validation component.

**Summary:**
* **Annotators:** 10
* **Response pairs:** 80
* **Scale:** 5-point ordinal consistency scale
* **Task Difficulty:** 2.6/5 (mean self-reported)
* Agreement and reliability statistics are reported in the manuscript.

---

## Licensing

### Code
Unless otherwise noted, source code in this repository is licensed under the **MIT License**. See `LICENSE`.

### Data, Text, Figures, and Documentation
Unless otherwise noted, non-code materials — including curated datasets, annotation materials, figures, and documentation — are licensed under **CC BY 4.0**. See `LICENSE-data.md`.

### Third-party Assets
Model weights and any third-party materials remain subject to their original upstream licenses and terms.

---

## Citation

If you use this repository, please cite the repository itself using `CITATION.cff`. 

If the article is published, please also cite the associated paper.

---

## Contact and Attribution

**Maintainer:** José Augusto de Lima Prestes

This repository is intended to support transparency, inspection, and reuse of the computational and validation components of the study.
