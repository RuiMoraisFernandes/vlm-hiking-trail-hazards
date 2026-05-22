# VLM Hiking Trail Hazard Assessment

Python scripts for evaluating Vision–Language Models in automated hiking trail hazard assessment from photographic imagery.

This repository contains the scripts used in the study:

> "Identification of Natural and Safety Hazards on Hiking Trails Using Vision–Language Models"

The workflow evaluates multiple VLMs under constrained and unconstrained prompting conditions for the identification of natural hazards, trail safety hazards, and overall danger assessment.

---

# Models Evaluated

- GPT-4o
- Gemini 3.1-flash-lite
- Mistral 3 Large
- Llama 4 Scout

Both constrained and unconstrained prompting strategies are included.

---

# Repository Structure

- `chatGPTConstrained.py` → GPT-4o constrained prompting
- `chatGPTUnconstrained.py` → GPT-4o unconstrained prompting
- `geminiConstrained.py` → Gemini constrained prompting
- `geminiUnconstrained.py` → Gemini unconstrained prompting
- `llamaConstrained.py` → Llama constrained prompting
- `llamaUnconstrained.py` → Llama unconstrained prompting
- `mistralConstrained.py` → Mistral constrained prompting
- `mistralUnconstrained.py` → Mistral unconstrained prompting

---

Extract the photoDataset.zip. The three .zip parts are required. 
The accompaning KMZ shows the location of the photos. Keep the KMZ exactly one folder above the photo folder in order to see the photo inside GoogleEarth.

# Requirements

Python 3.10 or newer is recommended.

Required Python packages:

```
openai
google-genai
groq
mistralai
pillow
pydantic

```
Replace the following block near the top of every script with your own information

```
API_KEY = "YOUR_API_KEY
IMAGE_FOLDER = r"IMAGE_INPUT_FOLDER"
EXPORT_FOLDER = r"JSON_OUTPUT_FOLDER"
```

Replace the output file name at the bottom of each script
`with open(os.path.join(EXPORT_FOLDER, "FILE_NAME.json"), "w") as f:`

---

# Notes

> API quotas and rate limits may affect execution speed.
> Results may vary slightly over time due to provider-side model updates.
> Some scripts include cooldown delays to reduce rate-limit errors.



