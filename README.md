# VLM Hiking Trail Hazard Assessment

The following are Python scripts and supporting data for the evaluation of Vision-Language Models (VLMs) for visual identification of natural and trail-safety hazards in hiking-trail photographs.

This repository accompanies the manuscript:

**Assessing Natural and Trail Safety Hazards on Hiking Trails Using Vision-Language Models**

The study evaluates four VLMs using two prompt formulations (constrained and unconstrained) across repeated API runs.

The scripts support:

- Natural-hazard identification
- Trail-safety hazard identification
- Overall danger classification
- Trail/location prediction
- Comparison with consensus reference annotations
- Calculation of hazard-detection performance metrics
- Recording of API response latency and run metadata

---

# Models Evaluated

The study evaluates:

- **OpenAI GPT-4o**
- **Google Gemini 3.1 Flash-Lite**
- **Mistral Large 3**
- **Meta Llama 4 Scout**

Each model is evaluated using both **constrained** and **unconstrained** prompt formulations.

The exact model identifiers used in the experiments are specified inside the corresponding scripts.

---

# Repository Structure

## Model Evaluation Scripts

Evaluation scripts are located in `/scripts`.

### Constrained

- `chatGPTConstrained.py`
- `geminiConstrained.py`
- `llamaConstrained.py`
- `mistralConstrained.py`

### Unconstrained

- `chatGPTUnconstrained.py`
- `geminiUnconstrained.py`
- `llamaUnconstrained.py`
- `mistralUnconstrained.py`

Each script contains the model-specific API configuration, prompt, image preprocessing, output handling, normalization/scoring procedures, and export routines used for that experimental condition.

---

# Dataset

The dataset contains **50 hiking-trail photographs** collected on Madeira Island, Portugal.

Images are provided in `/photoDataset`.

The dataset includes:

- Images containing natural hazards
- Images containing trail-safety hazards
- Control images without reference hazards

A reference annotation file provides the consensus hazard labels used for model evaluation.

A companion **KMZ file** provides the spatial locations associated with the photographs.

To visualize the photographs in Google Earth using the supplied KMZ structure, retain the relative file organization provided in the repository.

---

# What the Scripts Do

Each model script follows the same general workflow:

1. Load the trail photographs.
2. Preprocess the images.
3. Submit each processed image to the selected VLM API.
4. Record the raw model response and run metadata.
5. Save raw results before evaluation/scoring.
6. Compare hazard predictions with the consensus reference annotations.
7. Calculate TP, FP and FN counts.
8. Calculate evaluation metrics.
9. Export detailed JSON and Excel results.

The scripts also retain model outputs for danger classification and trail/location prediction.

Some implementation details differ between providers because their APIs and response formats are different.

---

# Evaluation Metrics

Hazard-identification performance is calculated from pooled TP, FP and FN counts.

The scripts report:

- **Recall**
- **Precision**
- **False Discovery Rate (FDR)**
- **F1 score**
- **Reporting Factor (RF)**

Additional outputs include class-level statistics, control-image false-positive information, and bootstrap confidence intervals.

Unconstrained scripts also record model expressions that cannot be mapped to the predefined hazard taxonomy. These are retained for inspection and scored according to the normalization procedure implemented in the corresponding script.

---

# Computational Requirements

No local GPU is required because model inference is performed through provider APIs.

Recommended requirements:

- Python 3.x
- Standard desktop or laptop computer
- Stable internet connection
- Valid API credentials for the relevant provider
- Sufficient API quota for the requested model

---

# Installation

Clone the repository:

```bash
git clone https://github.com/RuiMoraisFernandes/vlm-hiking-trail-hazards.git
cd vlm-hiking-trail-hazards
