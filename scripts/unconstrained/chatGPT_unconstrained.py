import os
import re
import json
import time
import base64
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from openai import OpenAI
from pydantic import BaseModel
import importlib.metadata


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Set it as an environment variable before running the script."
    )

IMAGE_FOLDER = Path(r"YOUR_PHOTO_PATH")
PREPROCESSED_FOLDER = Path(r"YOUR_PHOTO_PREPROCESS_PATH")
EXPORT_FOLDER = Path(r"YOUR_OUTPUT_FOLDER")

GROUND_TRUTH_EXCEL = Path(r"YOUR_REFERENCE_ANNOTATION")
GROUND_TRUTH_SHEET = 0  # sheet name or sheet index


# Ground-truth Excel column names
GT_FILENAME_COL = "filename"
GT_NATURAL_COL = "natural_hazards"
GT_SAFETY_COL = "trail_problems"
GT_DANGER_COL = "danger"


# Experiment identification
MODEL_ID = "gpt-4o-2024-08-06"
PROVIDER = "OpenAI"
PROMPT_TYPE = "unconstrained"
RUN_ID = 1
TEMPERATURE = 0.0


MAX_IMAGE_DIMENSION = 1500
JPEG_QUALITY = 85
PRESERVE_EXIF = True

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


# Workflow switches
# Set RUN_MODEL_INFERENCE=False if you already have the RAW JSON and
# only want to rerun normalization/scoring/statistics.
RUN_MODEL_INFERENCE = True
RUN_SCORING = True


# Output files - Customizable
RAW_JSON_FILE = EXPORT_FOLDER / f"gpt4o_unconstrained_run{RUN_ID}_RAW.json"
SCORED_JSON_FILE = EXPORT_FOLDER / f"gpt4o_unconstrained_run{RUN_ID}_SCORED.json"
STATISTICS_JSON_FILE = EXPORT_FOLDER / f"gpt4o_unconstrained_run{RUN_ID}_STATISTICS.json"
STATISTICS_EXCEL_FILE = EXPORT_FOLDER / f"gpt4o_unconstrained_run{RUN_ID}_STATISTICS.xlsx"


# ============================================================
# 2. EXPERIMENT TAXONOMY
# ============================================================

NATURAL_CLASSES = [
    "Wildfire",
    "Landslides/Rockfalls",
    "Flooding",
    "Solar exposure",
    "Volcanic activity",
    "Fog",
    "Wind",
    "Snow/Avalanche",
]

SAFETY_CLASSES = [
    "Steep path",
    "Steep slopes on side(s)",
    "Narrow path",
    "Damaged/irregular path",
    "Muddy/wet path",
    "Visitor over presence",
    "Broken/missing guardrails",
    "Faded/missing trail markers",
    "Damaged stairs/boardwalks",
    "Erosion",
    "Overhanging branches",
    "Exposed stones/roots",
    "Dangerous flora/fauna",
]


# ============================================================
# 3. GROUND-TRUTH NORMALIZATION
#
# These dictionaries are STRICT.

# Unknown ground-truth terms cause an error.
# ============================================================

GROUND_TRUTH_NATURAL_ALIASES = {
    # Canonical labels
    "wildfire": "Wildfire",
    "landslides/rockfalls": "Landslides/Rockfalls",
    "flooding": "Flooding",
    "solar exposure": "Solar exposure",
    "volcanic activity": "Volcanic activity",
    "fog": "Fog",
    "wind": "Wind",
    "snow/avalanche": "Snow/Avalanche",

    # Terminology actually used in your Excel
    "rockfall": "Landslides/Rockfalls",
    "rockfalls": "Landslides/Rockfalls",
    "landslide": "Landslides/Rockfalls",
    "landslides": "Landslides/Rockfalls",
}


GROUND_TRUTH_SAFETY_ALIASES = {
    # Canonical labels
    "steep path": "Steep path",
    "steep slopes on side(s)": "Steep slopes on the side(s) of the paths",
    "narrow path": "Narrow path",
    "damaged/irregular path": "Damaged/irregular path",
    "muddy/wet path": "Muddy/wet path",
    "visitor over presence": "Visitor overpresence",
    "broken/missing guardrails": "Broken/missing guardrails",
    "faded/missing trail markers": "Faded/missing trail markers",
    "damaged stairs/boardwalks": "Damaged stairs/boardwalks",
    "erosion": "Erosion",
    "overhanging branches": "Overhanging branches",
    "exposed stones/roots": "Exposed stones/roots on the path",
    "dangerous flora/fauna": "Dangerous flora/fauna",

    # Terminology actually used in your Excel
    "steep slopes": "Steep slopes on the side(s) of the paths",
    "steep slopes on the side(s)": "Steep slopes on the side(s) of the paths",
    "steep slopes on the side(s) of the path": "Steep slopes on the side(s) of the paths",
    "steep slopes on the side(s) of the paths": "Steep slopes on the side(s) of the paths",
    "mud": "Muddy/wet path",
    "missing guardrails": "Broken/missing guardrails",
    "damaged guardrails": "Broken/missing guardrails",
    "visitor overpresence": "Visitor overpresence",
    "damaged path": "Damaged/irregular path",
    "irregular path": "Damaged/irregular path",
    "wet path": "Muddy/wet path",
    "damaged stairs": "Damaged stairs/boardwalks",
    "exposed rocks": "Exposed stones/roots on the path",
}


# ============================================================
# 4. MODEL-OUTPUT NORMALIZATION
# ============================================================

MODEL_NATURAL_ALIASES = {
    # Canonical labels
    "wildfire": "Wildfire",
    "landslides/rockfalls": "Landslides/Rockfalls",
    "flooding": "Flooding",
    "solar exposure": "Solar exposure",
    "volcanic activity": "Volcanic activity",
    "fog": "Fog",
    "wind": "Wind",
    "snow/avalanche": "Snow/Avalanche",

    # Explicit accepted synonyms
    "forest fire": "Wildfire",
    "brush fire": "Wildfire",
    "vegetation fire": "Wildfire",

    "rockfall": "Landslides/Rockfalls",
    "rockfalls": "Landslides/Rockfalls",
    "landslide": "Landslides/Rockfalls",
    "landslides": "Landslides/Rockfalls",
    "falling rocks": "Landslides/Rockfalls",
    "rock fall": "Landslides/Rockfalls",

    "flood": "Flooding",
    "flash flood": "Flooding",

    "sun exposure": "Solar exposure",
    "strong sunlight": "Solar exposure",
    "intense sunlight": "Solar exposure",

    "volcanic hazard": "Volcanic activity",

    "mist": "Fog",
    "dense fog": "Fog",
    "heavy fog": "Fog",

    "strong wind": "Wind",
    "strong winds": "Wind",
    "high winds": "Wind",

    "snow": "Snow/Avalanche",
    "avalanche": "Snow/Avalanche",
}


MODEL_SAFETY_ALIASES = {
    # Canonical labels
    "steep path": "Steep path",
    "steep slopes on side(s)": "Steep slopes on side(s)",
    "narrow path": "Narrow path",
    "damaged/irregular path": "Damaged/irregular path",
    "muddy/wet path": "Muddy/wet path",
    "visitor over presence": "Visitor over presence",
    "broken/missing guardrails": "Broken/missing guardrails",
    "faded/missing trail markers": "Faded/missing trail markers",
    "damaged stairs/boardwalks": "Damaged stairs/boardwalks",
    "erosion": "Erosion",
    "overhanging branches": "Overhanging branches",
    "exposed stones/roots": "Exposed stones/roots",
    "dangerous flora/fauna": "Dangerous flora/fauna",

    # Accepted model-output synonyms
    "steep slopes": "Steep slopes on side(s)",
    "steep slope": "Steep slopes on side(s)",
    "steep side slope": "Steep slopes on side(s)",
    "steep drop": "Steep slopes on side(s)",
    "steep trail": "Steep path",
    "steep incline": "Steep path",

    "narrow trail": "Narrow path",

    "damaged path": "Damaged/irregular path",
    "irregular path": "Damaged/irregular path",
    "damaged trail": "Damaged/irregular path",
    "uneven path": "Damaged/irregular path",
    "uneven trail": "Damaged/irregular path",

    "mud": "Muddy/wet path",
    "wet path": "Muddy/wet path",
    "wet trail": "Muddy/wet path",
    "wet surface": "Muddy/wet path",
    "muddy path": "Muddy/wet path",
    "muddy trail": "Muddy/wet path",
    "muddy surface": "Muddy/wet path",


    "visitor overpresence": "Visitor over presence",
    "crowding": "Visitor over presence",
    "overcrowding": "Visitor over presence",

    "missing guardrails": "Broken/missing guardrails",
    "damaged guardrails": "Broken/missing guardrails",
    "broken guardrail": "Broken/missing guardrails",
    "broken guardrails": "Broken/missing guardrails",
    "missing guardrail": "Broken/missing guardrails",
    "damaged guardrail": "Broken/missing guardrails",
    "no guardrail": "Broken/missing guardrails",
    "no guardrails": "Broken/missing guardrails",

    "missing trail marker": "Faded/missing trail markers",
    "missing trail markers": "Faded/missing trail markers",
    "faded trail marker": "Faded/missing trail markers",
    "faded trail markers": "Faded/missing trail markers",

    "damaged stairs": "Damaged stairs/boardwalks",
    "broken stairs": "Damaged stairs/boardwalks",
    "damaged boardwalk": "Damaged stairs/boardwalks",
    "broken boardwalk": "Damaged stairs/boardwalks",

    "trail erosion": "Erosion",
    "eroded trail": "Erosion",
    "eroded path": "Erosion",

    "low branches": "Overhanging branches",

    "exposed rocks": "Exposed stones/roots",
    "exposed roots": "Exposed stones/roots",
    "tree roots": "Exposed stones/roots",
    "exposed stones": "Exposed stones/roots",
    "loose stones": "Exposed stones/roots",

    "dangerous vegetation": "Dangerous flora/fauna",
    "dangerous wildlife": "Dangerous flora/fauna",
}


# ============================================================
# 5. DANGER EVALUATION TERMS
# ============================================================

EVALUATION_POSSIBILITIES = [
    "No danger",
    "Slightly dangerous",
    "Moderately dangerous",
    "Very dangerous",
]


# ============================================================
# 6. PROMPT
# ============================================================

CHECKLIST_PROMPT = f"""
Inspect the trail for natural hazards.
Inspect the trail for infrastructure or trail-safety problems.

For natural hazards and trail-safety problems, respond only with
the name of each hazard or problem. Add nothing else.

If multiple terms are identified in a category, separate them using ';'.
If none are identified, answer with 'None'.
Under no circumstance leave a field blank.

Identify which trail the photo belongs to.
If the file contains coordinates, use them for the location.
If coordinates are not available, identify the location visually.

Give your final evaluation of risk for visitors.
Use only one of these terms:
{", ".join(EVALUATION_POSSIBILITIES)}.

Base the evaluation not only on the number of hazards/problems,
but also on their severity.
""".strip()


SYSTEM_PROMPT = (
    "You are a trail safety inspector. "
    "Analyze images for natural hazards and trail infrastructure problems."
)


# ============================================================
# 7. STRUCTURED RESPONSE
# ============================================================

class AnalysisResult(BaseModel):
    trail_name: str
    natural_hazards: str
    trail_problems: str
    evaluation: Literal[
        "No danger",
        "Slightly dangerous",
        "Moderately dangerous",
        "Very dangerous",
    ]


# ============================================================
# 8. SETUP
# ============================================================

client = OpenAI(api_key=OPENAI_API_KEY)

PREPROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 9. GENERAL HELPERS
# ============================================================

def sha256_file(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    return sha.hexdigest()


def has_exif(img):
    try:
        return len(img.getexif()) > 0
    except Exception:
        return False


def has_gps_exif(img):
    try:
        exif = img.getexif()
        return 34853 in exif
    except Exception:
        return False


def atomic_save_json(data, path):
    """
    Save JSON via a temporary file to reduce the risk of corruption
    if execution is interrupted during writing.
    """
    temp_path = str(path) + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    os.replace(temp_path, path)


# ============================================================
# 10. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(source_path, destination_path, preserve_exif=True):
    """
    Standardize every model input to:
    - maximum dimension = 1500 px
    - JPEG
    - quality = 85

    EXIF is either preserved or removed explicitly.
    """
    with Image.open(source_path) as original:

        original_width, original_height = original.size
        original_has_exif = has_exif(original)
        original_has_gps = has_gps_exif(original)

        # Correct orientation according to EXIF.
        img = ImageOps.exif_transpose(original)

        # Retrieve EXIF after orientation correction.
        exif_bytes = None

        if preserve_exif:
            try:
                exif = img.getexif()
                if len(exif) > 0:
                    exif_bytes = exif.tobytes()
            except Exception:
                exif_bytes = None

        if img.mode != "RGB":
            img = img.convert("RGB")

        img.thumbnail(
            (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            Image.Resampling.LANCZOS
        )

        processed_width, processed_height = img.size

        save_parameters = {
            "format": "JPEG",
            "quality": JPEG_QUALITY,
        }

        if preserve_exif and exif_bytes:
            save_parameters["exif"] = exif_bytes

        img.save(destination_path, **save_parameters)

    # Verify resulting file.
    with Image.open(destination_path) as processed:
        processed_has_exif = has_exif(processed)
        processed_has_gps = has_gps_exif(processed)

    return {
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "original_has_exif": original_has_exif,
        "original_has_gps": original_has_gps,
        "processed_has_exif": processed_has_exif,
        "processed_has_gps": processed_has_gps,
    }


def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ============================================================
# 11. RAW MODEL INFERENCE
#
# IMPORTANT:
# NO SCORING OR GROUND-TRUTH COMPARISON OCCURS HERE.
# RAW JSON IS SAVED BEFORE ANY CALCULATIONS.
# ============================================================

def run_inference():

    experiment_start = time.time()

    experiment_metadata = {
        "provider": PROVIDER,
        "model_requested": MODEL_ID,
        "prompt_type": PROMPT_TYPE,
        "run_id": RUN_ID,
        "temperature": TEMPERATURE,
        "max_image_dimension": MAX_IMAGE_DIMENSION,
        "jpeg_quality": JPEG_QUALITY,
        "preserve_exif": PRESERVE_EXIF,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": CHECKLIST_PROMPT,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "openai_package_version": importlib.metadata.version("openai"),
        "pillow_version": importlib.metadata.version("Pillow"),
        "pandas_version": importlib.metadata.version("pandas"),
    }

    records = []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    image_files = sorted([
        filename
        for filename in os.listdir(IMAGE_FOLDER)
        if filename.lower().endswith(valid_extensions)
    ])

    for index, filename in enumerate(image_files, start=1):

        original_path = IMAGE_FOLDER / filename

        processed_filename = Path(filename).stem + ".jpg"
        processed_path = PREPROCESSED_FOLDER / processed_filename

        print(
            f"[{index}/{len(image_files)}] "
            f"Processing {filename}"
        )

        try:
            preprocessing_info = preprocess_image(
                original_path,
                processed_path,
                preserve_exif=PRESERVE_EXIF
            )

            base64_image = encode_image(processed_path)

            # API response latency starts immediately before the API call.
            request_start = time.perf_counter()

            response = client.beta.chat.completions.parse(
                model=MODEL_ID,
                temperature=TEMPERATURE,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": CHECKLIST_PROMPT,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],

                response_format=AnalysisResult,
            )

            # Stop immediately after the complete API response is returned.
            request_end = time.perf_counter()

            api_latency = round(
                request_end - request_start,
                3
            )

            analysis = response.choices[0].message.parsed
            usage = getattr(response, "usage", None)

            record = {
                "status": "success",
                "original_filename": filename,
                "processed_filename": processed_filename,

                "original_sha256": sha256_file(original_path),
                "processed_sha256": sha256_file(processed_path),

                **preprocessing_info,

                "provider": PROVIDER,
                "requested_model": MODEL_ID,
                "returned_model": getattr(response, "model", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "response_id": getattr(response, "id", None),

                "temperature": TEMPERATURE,
                "prompt_type": PROMPT_TYPE,
                "run_id": RUN_ID,

                # RAW MODEL OUTPUT
                "trail_name": analysis.trail_name,
                "natural_hazards": analysis.natural_hazards,
                "trail_problems": analysis.trail_problems,
                "evaluation": analysis.evaluation,

                "api_response_latency_seconds": api_latency,

                "prompt_tokens": (
                    usage.prompt_tokens
                    if usage else None
                ),
                "completion_tokens": (
                    usage.completion_tokens
                    if usage else None
                ),
                "total_tokens": (
                    usage.total_tokens
                    if usage else None
                ),
            }

            records.append(record)

            print(
                f"  ✓ {filename}: "
                f"{api_latency:.2f} seconds API latency"
            )

        except Exception as exc:

            record = {
                "status": "error",
                "original_filename": filename,
                "provider": PROVIDER,
                "requested_model": MODEL_ID,
                "prompt_type": PROMPT_TYPE,
                "run_id": RUN_ID,
                "error": str(exc),
            }

            records.append(record)

            print(
                f"  ✗ ERROR: {filename}: {exc}"
            )

        raw_package = {
            "experiment_metadata": experiment_metadata,
            "results": records,
        }

        atomic_save_json(
            raw_package,
            RAW_JSON_FILE
        )

    experiment_metadata["total_experiment_seconds"] = round(
        time.time() - experiment_start,
        2
    )

    raw_package = {
        "experiment_metadata": experiment_metadata,
        "results": records,
    }

    atomic_save_json(
        raw_package,
        RAW_JSON_FILE
    )

    print(
        f"\nRAW model output saved to:\n"
        f"{RAW_JSON_FILE}"
    )


# ============================================================
# 12. TEXT NORMALIZATION
# ============================================================

def clean_term(term):

    term = str(term).strip().lower()

    term = term.replace("–", "-")
    term = term.replace("—", "-")

    term = re.sub(
        r"^[\-\*\u2022]+\s*",
        "",
        term
    )

    term = re.sub(
        r"\s+",
        " ",
        term
    )

    term = term.strip(" .,:")

    return term


def split_terms(text):
    """
    The prompt explicitly requests ';' as separator.
    Newlines are also accepted in case the model does not
    perfectly follow formatting instructions.
    """

    if text is None:
        return []

    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    if clean_term(text) in {
        "none",
        "no",
        "no hazards",
        "no hazard",
        "no problems",
        "no problem",
        "n/a",
    }:
        return []

    pieces = re.split(
        r"[;\n]+",
        text
    )

    return [
        clean_term(piece)
        for piece in pieces
        if clean_term(piece)
    ]


def normalize_prediction(raw_text, model_aliases):
    """
    Normalize unconstrained MODEL OUTPUT.

    Returns:
    - normalized class set
    - list of unmapped raw terms

    Unknown model expressions are assigned to "Other".
    """

    raw_terms = split_terms(raw_text)

    normalized = set()
    unmapped = []

    for term in raw_terms:

        if term in model_aliases:
            normalized.add(
                model_aliases[term]
            )
        else:
            normalized.add("Other")
            unmapped.append(term)

    return normalized, unmapped


def normalize_ground_truth(raw_text, ground_truth_aliases, allowed_classes):
    """
    Normalize HUMAN REFERENCE / GROUND TRUTH.

    Unlike predictions, unknown ground-truth terms cause an error.
    This prevents accidental semantic guessing in the reference data.
    """

    terms = split_terms(raw_text)

    normalized = set()

    for term in terms:

        if term in ground_truth_aliases:
            canonical = ground_truth_aliases[term]
        else:
            raise ValueError(
                "Unknown ground-truth term: "
                f"'{term}'. "
                "Correct the Excel file or add it deliberately to the "
                "GROUND_TRUTH_*_ALIASES dictionary."
            )

        if canonical not in allowed_classes:
            raise ValueError(
                f"Ground-truth term '{canonical}' "
                "is not a recognized canonical class."
            )

        normalized.add(canonical)

    return normalized


# ============================================================
# 13. METRIC CALCULATION
# ============================================================

def safe_divide(numerator, denominator):

    if denominator == 0:
        return None

    return numerator / denominator


def metrics_from_counts(tp, fp, fn):

    recall = safe_divide(
        tp,
        tp + fn
    )

    precision = safe_divide(
        tp,
        tp + fp
    )

    fdr = safe_divide(
        fp,
        tp + fp
    )

    rf = safe_divide(
        tp + fp,
        tp + fn
    )

    if (
        precision is None
        or recall is None
        or precision + recall == 0
    ):
        f1 = None
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Recall": recall,
        "Precision": precision,
        "FDR": fdr,
        "F1": f1,
        "RF": rf,
    }


# ============================================================
# 14. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_confidence_intervals(
    scored_rows,
    bootstrap_samples=2000,
    seed=42
):

    rng = np.random.default_rng(seed)

    n = len(scored_rows)

    if n == 0:
        return {
            metric: {
                "CI95_lower": None,
                "CI95_upper": None,
            }
            for metric in ["Recall", "Precision", "FDR", "F1", "RF"]
        }

    metric_names = [
        "Recall",
        "Precision",
        "FDR",
        "F1",
        "RF",
    ]

    bootstrap_values = {
        metric: []
        for metric in metric_names
    }

    for _ in range(bootstrap_samples):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        tp = sum(
            scored_rows[i]["TP"]
            for i in indices
        )

        fp = sum(
            scored_rows[i]["FP"]
            for i in indices
        )

        fn = sum(
            scored_rows[i]["FN"]
            for i in indices
        )

        metrics = metrics_from_counts(
            tp,
            fp,
            fn
        )

        for metric in metric_names:

            value = metrics[metric]

            if value is not None:
                bootstrap_values[metric].append(value)

    confidence_intervals = {}

    for metric, values in bootstrap_values.items():

        if not values:

            confidence_intervals[metric] = {
                "CI95_lower": None,
                "CI95_upper": None,
            }

        else:

            confidence_intervals[metric] = {
                "CI95_lower": float(
                    np.percentile(values, 2.5)
                ),
                "CI95_upper": float(
                    np.percentile(values, 97.5)
                ),
            }

    return confidence_intervals


# ============================================================
# 15. SCORE ONE DOMAIN
# ============================================================

def score_domain(
    merged_dataframe,
    prediction_column,
    ground_truth_column,
    model_aliases,
    ground_truth_aliases,
    allowed_classes,
    domain_name
):

    scored_rows = []
    all_unmapped = []

    for _, row in merged_dataframe.iterrows():

        filename = row["original_filename"]

        # MODEL OUTPUT uses model-output aliases.
        prediction_set, unmapped = normalize_prediction(
            row[prediction_column],
            model_aliases
        )

        # HUMAN REFERENCE uses ground-truth aliases.
        reference_set = normalize_ground_truth(
            row[ground_truth_column],
            ground_truth_aliases,
            allowed_classes
        )

        tp_classes = prediction_set & reference_set
        fp_classes = prediction_set - reference_set
        fn_classes = reference_set - prediction_set

        scored = {
            "filename": filename,
            "domain": domain_name,

            "prediction_raw": row[prediction_column],

            "prediction_normalized": (
                "; ".join(sorted(prediction_set))
                if prediction_set
                else "None"
            ),

            "reference_raw": row[ground_truth_column],

            "reference_normalized": (
                "; ".join(sorted(reference_set))
                if reference_set
                else "None"
            ),

            "TP_classes": (
                "; ".join(sorted(tp_classes))
                if tp_classes
                else "None"
            ),

            "FP_classes": (
                "; ".join(sorted(fp_classes))
                if fp_classes
                else "None"
            ),

            "FN_classes": (
                "; ".join(sorted(fn_classes))
                if fn_classes
                else "None"
            ),

            "TP": len(tp_classes),
            "FP": len(fp_classes),
            "FN": len(fn_classes),

            "is_control": len(reference_set) == 0,

            "control_has_false_positive": (
                len(reference_set) == 0
                and len(prediction_set) > 0
            ),
        }

        scored_rows.append(scored)

        for term in unmapped:

            all_unmapped.append({
                "filename": filename,
                "domain": domain_name,
                "raw_term": term,
                "assigned_class": "Other",
            })

    # --------------------------------------------------------
    # MICRO / POOLED STATISTICS
    # --------------------------------------------------------

    total_tp = sum(
        row["TP"]
        for row in scored_rows
    )

    total_fp = sum(
        row["FP"]
        for row in scored_rows
    )

    total_fn = sum(
        row["FN"]
        for row in scored_rows
    )

    overall = metrics_from_counts(
        total_tp,
        total_fp,
        total_fn
    )

    overall["domain"] = domain_name
    overall["n_images"] = len(scored_rows)

    # --------------------------------------------------------
    # CONTROL IMAGE STATISTICS
    # --------------------------------------------------------

    controls = [
        row
        for row in scored_rows
        if row["is_control"]
    ]

    if controls:

        controls_with_fp = sum(
            row["control_has_false_positive"]
            for row in controls
        )

        total_control_fp = sum(
            row["FP"]
            for row in controls
        )

        overall["control_images"] = len(controls)
        overall["controls_with_false_positive"] = controls_with_fp

        overall["control_false_positive_image_rate"] = (
            controls_with_fp
            / len(controls)
        )

        overall["control_clear_rate"] = (
            1
            - controls_with_fp
            / len(controls)
        )

        overall["false_positive_classes_on_controls"] = total_control_fp

        overall["mean_false_positive_classes_per_control"] = (
            total_control_fp
            / len(controls)
        )

    else:

        overall["control_images"] = 0
        overall["controls_with_false_positive"] = 0
        overall["control_false_positive_image_rate"] = None
        overall["control_clear_rate"] = None
        overall["false_positive_classes_on_controls"] = 0
        overall["mean_false_positive_classes_per_control"] = None

    # --------------------------------------------------------
    # BOOTSTRAP CI
    # --------------------------------------------------------

    ci = bootstrap_confidence_intervals(
        scored_rows,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED
    )

    for metric, interval in ci.items():

        overall[f"{metric}_CI95_lower"] = interval["CI95_lower"]
        overall[f"{metric}_CI95_upper"] = interval["CI95_upper"]

    # --------------------------------------------------------
    # CLASS-WISE STATISTICS
    # --------------------------------------------------------

    class_rows = []

    evaluation_classes = allowed_classes + ["Other"]

    for hazard_class in evaluation_classes:

        class_tp = 0
        class_fp = 0
        class_fn = 0

        for _, source_row in merged_dataframe.iterrows():

            pred_set, _ = normalize_prediction(
                source_row[prediction_column],
                model_aliases
            )

            ref_set = normalize_ground_truth(
                source_row[ground_truth_column],
                ground_truth_aliases,
                allowed_classes
            )

            predicted = hazard_class in pred_set
            reference = hazard_class in ref_set

            if predicted and reference:
                class_tp += 1

            elif predicted and not reference:
                class_fp += 1

            elif not predicted and reference:
                class_fn += 1

        class_metrics = metrics_from_counts(
            class_tp,
            class_fp,
            class_fn
        )

        class_metrics["domain"] = domain_name
        class_metrics["class"] = hazard_class

        class_rows.append(class_metrics)

    return (
        scored_rows,
        overall,
        class_rows,
        all_unmapped
    )


# ============================================================
# 16. DANGER EVALUATION / CONFUSION MATRIX
# ============================================================

def normalize_danger(value, source_name):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        raise ValueError(f"Missing danger value in {source_name}.")
    cleaned = clean_term(value)
    lookup = {clean_term(label): label for label in EVALUATION_POSSIBILITIES}
    if cleaned not in lookup:
        raise ValueError(
            f"Unknown danger label '{value}' in {source_name}. "
            f"Expected one of: {EVALUATION_POSSIBILITIES}"
        )
    return lookup[cleaned]


def calculate_danger_results(merged):
    rows = []
    for _, row in merged.iterrows():
        reference = normalize_danger(row["gt_danger"], "human reference")
        prediction = normalize_danger(row["evaluation"], "model output")
        rows.append({
            "filename": row["original_filename"],
            "reference_danger": reference,
            "predicted_danger": prediction,
            "correct": reference == prediction,
        })

    comparison_df = pd.DataFrame(rows)
    counts = pd.crosstab(
        comparison_df["reference_danger"],
        comparison_df["predicted_danger"],
        dropna=False,
    ).reindex(
        index=EVALUATION_POSSIBILITIES,
        columns=EVALUATION_POSSIBILITIES,
        fill_value=0,
    )
    counts.index.name = "Human reference"
    counts.columns.name = "GPT-4o prediction"

    percentages = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    percentages = percentages.fillna(0.0)
    percentages.index.name = "Human reference"
    percentages.columns.name = "GPT-4o prediction"

    exact_accuracy = comparison_df["correct"].mean() if len(comparison_df) else None
    summary_df = pd.DataFrame([{
        "n_images": len(comparison_df),
        "exact_matches": int(comparison_df["correct"].sum()),
        "exact_accuracy": exact_accuracy,
    }])
    return comparison_df, counts, percentages, summary_df


# ============================================================
# 17. LOAD RAW JSON AND GROUND TRUTH
# ============================================================

def score_saved_raw_results():

    print("\nStarting scoring stage...")

    # -----------------------------
    # Load RAW model output
    # -----------------------------

    if not RAW_JSON_FILE.exists():
        raise FileNotFoundError(
            f"RAW JSON does not exist:\n{RAW_JSON_FILE}"
        )

    with open(
        RAW_JSON_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        raw_package = json.load(f)

    raw_results = raw_package["results"]

    successful_results = [
        row
        for row in raw_results
        if row.get("status") == "success"
    ]

    if not successful_results:
        raise ValueError(
            "No successful model results were found in the RAW JSON."
        )

    model_df = pd.DataFrame(
        successful_results
    )

    # -----------------------------
    # Load ground truth Excel
    # -----------------------------

    ground_truth_df = pd.read_excel(
        GROUND_TRUTH_EXCEL,
        sheet_name=GROUND_TRUTH_SHEET
    )

    required_columns = {
        GT_FILENAME_COL,
        GT_NATURAL_COL,
        GT_SAFETY_COL,
        GT_DANGER_COL,
    }

    missing_columns = (
        required_columns
        - set(ground_truth_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Ground-truth Excel is missing columns: "
            f"{missing_columns}"
        )

    # -----------------------------
    # Normalized filename join keys
    # -----------------------------

    # -----------------------------
    # Normalized filename join keys
    # -----------------------------

    model_df["_join_filename"] = (
        model_df["original_filename"]
        .astype(str)
        .str.strip()
        .apply(lambda x: Path(x).stem.lower())
    )

    ground_truth_df["_join_filename"] = (
        ground_truth_df[GT_FILENAME_COL]
        .astype(str)
        .str.strip()
        .apply(lambda x: Path(x).stem.lower())
    )

    print("\nFirst model join keys:")
    print(
        model_df[
            ["original_filename", "_join_filename"]
        ].head(10).to_string(index=False)
    )

    print("\nFirst ground-truth join keys:")
    print(
        ground_truth_df[
            [GT_FILENAME_COL, "_join_filename"]
        ].head(10).to_string(index=False)
    )

    # -----------------------------
    # Check duplicate filenames
    # -----------------------------

    duplicate_gt = ground_truth_df["_join_filename"].duplicated(
        keep=False
    )

    if duplicate_gt.any():

        duplicated_names = (
            ground_truth_df.loc[
                duplicate_gt,
                GT_FILENAME_COL
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate filenames in ground-truth Excel: "
            f"{duplicated_names}"
        )

    duplicate_model = model_df["_join_filename"].duplicated(
        keep=False
    )

    if duplicate_model.any():

        duplicated_names = (
            model_df.loc[
                duplicate_model,
                "original_filename"
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate filenames in model RAW JSON: "
            f"{duplicated_names}"
        )

    # -----------------------------
    # Rename ground-truth columns
    # BEFORE merge, so there is no
    # collision with model columns.
    # -----------------------------

    ground_truth_for_merge = ground_truth_df[
        [
            "_join_filename",
            GT_NATURAL_COL,
            GT_SAFETY_COL,
            GT_DANGER_COL,
        ]
    ].rename(
        columns={
            GT_NATURAL_COL: "gt_natural_hazards",
            GT_SAFETY_COL: "gt_trail_problems",
            GT_DANGER_COL: "gt_danger",
        }
    )

    # -----------------------------
    # Join predictions to reference
    # -----------------------------

    merged = model_df.merge(
        ground_truth_for_merge,
        on="_join_filename",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    # -----------------------------
    # Check unmatched model files
    # -----------------------------

    unmatched = merged["_merge"] != "both"

    if unmatched.any():

        missing_names = (
            merged.loc[
                unmatched,
                "original_filename"
            ]
            .tolist()
        )

        raise ValueError(
            "These model images have no matching ground-truth row:\n"
            f"{missing_names}"
        )

    merged = merged.drop(columns=["_merge"])

    # -----------------------------
    # Check Excel rows with no model result
    # -----------------------------

    model_keys = set(model_df["_join_filename"])
    gt_keys = set(ground_truth_df["_join_filename"])

    gt_without_model = sorted(gt_keys - model_keys)

    if gt_without_model:
        print(
            "\nWARNING: The following ground-truth rows have no successful "
            "model result and will not be scored:"
        )

        for filename in gt_without_model:
            print(f"  - {filename}")

    # ========================================================
    # NATURAL HAZARDS
    # ========================================================

    (
        natural_scored,
        natural_overall,
        natural_classwise,
        natural_unmapped,
    ) = score_domain(
        merged_dataframe=merged,

        prediction_column="natural_hazards",
        ground_truth_column="gt_natural_hazards",

        model_aliases=MODEL_NATURAL_ALIASES,
        ground_truth_aliases=GROUND_TRUTH_NATURAL_ALIASES,

        allowed_classes=NATURAL_CLASSES,
        domain_name="Natural hazards",
    )

    # ========================================================
    # TRAIL-SAFETY PROBLEMS
    # ========================================================

    (
        safety_scored,
        safety_overall,
        safety_classwise,
        safety_unmapped,
    ) = score_domain(
        merged_dataframe=merged,

        prediction_column="trail_problems",
        ground_truth_column="gt_trail_problems",

        model_aliases=MODEL_SAFETY_ALIASES,
        ground_truth_aliases=GROUND_TRUTH_SAFETY_ALIASES,

        allowed_classes=SAFETY_CLASSES,
        domain_name="Trail safety",
    )

    # ========================================================
    # DANGER CLASSIFICATION
    # ========================================================

    danger_comparison, danger_counts, danger_percent, danger_summary = (
        calculate_danger_results(merged)
    )

    # ========================================================
    # COMBINE OUTPUT
    # ========================================================

    scored_rows = (
        natural_scored
        + safety_scored
    )

    overall_statistics = [
        natural_overall,
        safety_overall,
    ]

    classwise_statistics = (
        natural_classwise
        + safety_classwise
    )

    unmapped_terms = (
        natural_unmapped
        + safety_unmapped
    )

    # ========================================================
    # SAVE SCORED JSON
    # ========================================================

    scored_package = {
        "source_raw_json": str(RAW_JSON_FILE),
        "ground_truth_excel": str(GROUND_TRUTH_EXCEL),

        "scoring_method": (
            "Micro-averaged pooled TP/FP/FN counts across images"
        ),

        "normalization_policy": {
            "ground_truth": (
                "Strict explicit mapping. Unknown reference terms raise an error."
            ),
            "model_output": (
                "Explicit model synonym mapping. Unmapped model terms are assigned to Other."
            ),
        },

        "scored_results": scored_rows,
        "danger_comparison": danger_comparison.to_dict(orient="records"),
    }

    atomic_save_json(
        scored_package,
        SCORED_JSON_FILE
    )

    # ========================================================
    # SAVE STATISTICS JSON
    # ========================================================

    statistics_package = {
        "overall_statistics": overall_statistics,
        "classwise_statistics": classwise_statistics,
        "unmapped_terms": unmapped_terms,
        "danger_summary": danger_summary.to_dict(orient="records"),
        "danger_confusion_counts": danger_counts.to_dict(),
        "danger_confusion_percent": danger_percent.to_dict(),
    }

    atomic_save_json(
        statistics_package,
        STATISTICS_JSON_FILE
    )

    # ========================================================
    # SAVE EXCEL WORKBOOK
    # ========================================================

    scored_df = pd.DataFrame(
        scored_rows
    )

    overall_df = pd.DataFrame(
        overall_statistics
    )

    classwise_df = pd.DataFrame(
        classwise_statistics
    )

    unmapped_df = pd.DataFrame(
        unmapped_terms
    )

    if unmapped_df.empty:
        unmapped_df = pd.DataFrame(
            columns=[
                "filename",
                "domain",
                "raw_term",
                "assigned_class",
            ]
        )

    metadata_df = pd.DataFrame([
        raw_package["experiment_metadata"]
    ]).T.reset_index()

    metadata_df.columns = [
        "parameter",
        "value"
    ]

    with pd.ExcelWriter(
        STATISTICS_EXCEL_FILE,
        engine="openpyxl"
    ) as writer:

        overall_df.to_excel(
            writer,
            sheet_name="Overall_statistics",
            index=False
        )

        classwise_df.to_excel(
            writer,
            sheet_name="Classwise_statistics",
            index=False
        )

        scored_df.to_excel(
            writer,
            sheet_name="Per_image_scoring",
            index=False
        )

        unmapped_df.to_excel(
            writer,
            sheet_name="Unmapped_terms",
            index=False
        )

        inference_df = model_df.drop(columns=["_join_filename"], errors="ignore").copy()
        inference_df.to_excel(writer, sheet_name="Inference_results", index=False)

        timing_values = pd.to_numeric(
            inference_df.get("api_response_latency_seconds", pd.Series(dtype=float)),
            errors="coerce"
        ).dropna()
        timing_summary_df = pd.DataFrame([{
            "n_successful_images": len(inference_df),
            "mean_api_response_latency_seconds": timing_values.mean() if len(timing_values) else None,
            "sd_api_response_latency_seconds": timing_values.std(ddof=1) if len(timing_values) > 1 else None,
            "median_api_response_latency_seconds": timing_values.median() if len(timing_values) else None,
            "legacy_total_experiment_seconds": raw_package["experiment_metadata"].get("total_experiment_seconds"),
        }])
        timing_summary_df.to_excel(writer, sheet_name="Timing_summary", index=False)

        danger_summary.to_excel(writer, sheet_name="Danger_summary", index=False)
        danger_comparison.to_excel(writer, sheet_name="Danger_comparison", index=False)
        danger_counts.to_excel(writer, sheet_name="Danger_confusion_counts")
        danger_percent.to_excel(writer, sheet_name="Danger_confusion_percent")

        metadata_df.to_excel(
            writer,
            sheet_name="Run_metadata",
            index=False
        )

    print("\nScoring complete.")

    print(
        f"\nScored JSON:\n"
        f"{SCORED_JSON_FILE}"
    )

    print(
        f"\nStatistics JSON:\n"
        f"{STATISTICS_JSON_FILE}"
    )

    print(
        f"\nStatistics Excel workbook:\n"
        f"{STATISTICS_EXCEL_FILE}"
    )

    print("\nOverall results:")

    print(
        overall_df.to_string(
            index=False
        )
    )

    print("\nDanger confusion matrix (counts):")
    print(danger_counts.to_string())

    if unmapped_terms:
        print(
            "\nNOTE: Some unconstrained model terms were not explicitly mapped "
            "and were scored as 'Other'. Review the Unmapped_terms sheet."
        )


# ============================================================
# 17. MAIN
# ============================================================

if __name__ == "__main__":

    # STEP 1:
    # Inference -> RAW JSON.
    #
    # Set RUN_MODEL_INFERENCE=False if you already have the RAW JSON
    # and only want to adjust aliases / rerun scoring.
    if RUN_MODEL_INFERENCE:
        run_inference()

    # STEP 2:
    # Load the already-saved RAW JSON + Excel reference,
    # normalize, score, and calculate statistics.
    if RUN_SCORING:
        score_saved_raw_results()
