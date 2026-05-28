# VLM Hiking Trail Hazard Assessment

Python scripts and supporting material for the evaluation of Vision–Language Models (VLMs) in automated hiking trail hazard assessment from photographic imagery.

This repository accompanies the manuscript:

**Assessing Natural and Trail Safety Hazards on Hiking Trails Using Vision–Language Models**

The study evaluates the capability of multiple Vision–Language Models to identify:

* **Natural hazards** (e.g., fog, landslides, wildfire exposure, wind-related conditions)
* **Trail safety hazards** (e.g., damaged surfaces, erosion, degraded infrastructure)
* **Overall trail danger level**
* **Photo spatial location**

using photographic data collected across the official hiking trail network of Madeira Island.

---

# Models Evaluated

* **GPT-4o**
* **Gemini 3.1-flash-lite**
* **Mistral 3 Large**
* **Llama 4 Scout**

Each model was evaluated under both constrained and unconstrained prompting strategies.

---

# Repository Structure

## Model Evaluation Scripts

Scripts are found under /scripts

### Constrained

* `chatGPTConstrained.py`
* `geminiConstrained.py`
* `llamaConstrained.py`
* `mistralConstrained.py`

### Unconstrained

* `chatGPTUnconstrained.py`
* `geminiUnconstrained.py`
* `llamaUnconstrained.py`
* `mistralUnconstrained.py`

## Dataset

The dataset consists of **50 hiking trail photographs** collected across Madeira Island.
Photos are located on /photoDataset

It includes:

* Images containing identifiable natural hazards
* Images containing identifiable trail safety hazards
* Control images without identifiable hazards

A companion **KMZ file** provides the spatial location of all photographs.

To visualize images inside Google Earth, keep the KMZ file **one directory level above** the image folder.

---

# Computational Requirements

The scripts were tested on a standard laptop environment.

## Minimum Requirements

* Standard desktop or laptop computer
* Stable internet connection
* Python 3.x
* Valid API access credentials for the corresponding VLM providers

No GPU or specialised hardware is required.

Inference is performed remotely through provider APIs.

---

# Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

If needed, install manually:

```bash
pip install openai google-genai groq mistralai pillow pydantic
```

---

# Required Python Packages

* `openai`
* `google-genai`
* `groq`
* `mistralai`
* `pillow`
* `pydantic`

---

# Configuration

Before running any script, replace the configuration block near the top of each file:

```python
API_KEY = "YOUR_API_KEY"
IMAGE_FOLDER = r"IMAGE_INPUT_FOLDER"
EXPORT_FOLDER = r"JSON_OUTPUT_FOLDER"
```

Replace:

* `YOUR_API_KEY` with your model API credential
* `IMAGE_INPUT_FOLDER` with the dataset directory
* `JSON_OUTPUT_FOLDER` with the desired output directory

---

# Output File Configuration

At the end of each script, define the output filename:

```python
with open(os.path.join(EXPORT_FOLDER, "FILE_NAME.json"), "w") as f:
```

Use distinct filenames for each experimental condition.

Example:

* `gemini_constrained.json`
* `gpt4o_unconstrained.json`

---

# Quick Start

## 1. Clone repository

```bash
git clone https://github.com/RuiMoraisFernandes/vlm-hiking-trail-hazards.git
cd vlm-hiking-trail-hazards
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```
Alternatively, install dependencies manually as per the intructions above.
 
## 3. Download dataset

Place all images in the designated input folder.

## 4. Insert API credentials

Edit the configuration block in the desired script.

## 5. Run evaluation

Example:

```bash
python geminiConstrained.py
```

## 6. Inspect outputs

Results are exported as structured JSON files.
The JSON can then be imported into other software (e.g. MS Excel)

---

# Inputs

Each script requires:

* JPEG image files
* Valid API key
* Input directory path
* Output directory path

---

# Outputs

Each script exports structured JSON containing:

* Image identifier
* Detected natural hazards
* Detected trail safety hazards
* Danger evaluation
* Processing time
* Inferred location (when applicable)

---

# Expected Behaviour

Scripts sequentially process all images in the specified folder.

Execution time depends on:

* API response latency
* Provider-side rate limits
* Temporary service quotas

Cooldown delays are included in some scripts to reduce rate-limit errors.

---

# Notes and Limitations

* API quotas may affect execution speed
* Provider-side model updates may slightly affect outputs over time
* Regional server routing may influence response variability
* Exact replication may depend on API version availability at execution time

---

# Data Availability

The image dataset and associated spatial reference files are provided in this repository under /photoDataset

---

# Code Availability

Source code is publicly available at:

https://github.com/RuiMoraisFernandes/vlm-hiking-trail-hazards

---

# License

This project is distributed under the **MIT License**.




