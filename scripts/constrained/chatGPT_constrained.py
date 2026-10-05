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
import importlib.metadata

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from openai import OpenAI
from pydantic import BaseModel


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

# SECURITY: set OPENAI_API_KEY as an environment variable.
OPENAI_API_KEY = "YOUR_API_KEY"
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Set it as an environment variable before running."
    )

IMAGE_FOLDER = Path(r"YOUR_PHOTO_PATH")
PREPROCESSED_FOLDER = Path(r"YOUR_PHOTO_PREPROCESS_PATH")
EXPORT_FOLDER = Path(r"YOUR_OUTPUT_FOLDER")

GROUND_TRUTH_EXCEL = Path(r"YOUR_REFERENCE_ANNOTATION")
GROUND_TRUTH_SHEET = 0  # sheet name or sheet index

# Ground-truth Excel columns.
GT_FILENAME_COL = "filename"
GT_NATURAL_COL = "natural_hazards"
GT_SAFETY_COL = "trail_problems"
GT_DANGER_COL = "danger"

# Experiment identification
MODEL_ID = "gpt-4o-2024-08-06"
PROVIDER = "OpenAI"
PROMPT_TYPE = "constrained"
RUN_ID = 1

# Inference settings
TEMPERATURE = 0.0

# Image preprocessing
MAX_IMAGE_DIMENSION = 1500
JPEG_QUALITY = 85
PRESERVE_EXIF = True

# True = preserve EXIF in resized JPEG if the original contains it.
# False = remove EXIF.
PRESERVE_EXIF = True

# Bootstrap confidence intervals
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42

# Workflow switches
# Set RUN_MODEL_INFERENCE=False if you already have the RAW JSON and
# only want to rerun normalization/scoring/statistics.
RUN_MODEL_INFERENCE = True
RUN_SCORING = True

# Output files - Customizable
RAW_JSON_FILE = EXPORT_FOLDER / f"gpt4o_constrained_run{RUN_ID}_RAW.json"
SCORED_JSON_FILE = EXPORT_FOLDER / f"gpt4o_constrained_run{RUN_ID}_SCORED.json"
STATISTICS_JSON_FILE = EXPORT_FOLDER / f"gpt4o_constrained_run{RUN_ID}_STATISTICS.json"
STATISTICS_EXCEL_FILE = EXPORT_FOLDER / f"gpt4o_constrained_run{RUN_ID}_STATISTICS.xlsx"


# ============================================================
# 2. CONSTRAINED TAXONOMY
# Exact labels supplied to the model and used for prediction scoring.
# ============================================================

NATURAL_HAZARDS_CHECKLIST = [
    "Wildfire",
    "Landslides/Rockfalls",
    "Flooding",
    "Solar exposure",
    "Volcanic activity",
    "Fog",
    "Snow/Avalanche",
    "Wind",
]

INFRASTRUCTURE_PROBLEMS_CHECKLIST = [
    "Steep path",
    "Steep slopes on the side(s) of the paths",
    "Narrow path",
    "Damaged/irregular path",
    "Muddy/wet path",
    "Visitor overpresence",
    "Broken/missing guardrails",
    "Faded/missing trail markers",
    "Damaged stairs/boardwalks",
    "Erosion",
    "Overhanging branches",
    "Exposed stones/roots on the path",
    "Dangerous flora/fauna",
]

NATURAL_CLASSES = NATURAL_HAZARDS_CHECKLIST.copy()
SAFETY_CLASSES = INFRASTRUCTURE_PROBLEMS_CHECKLIST.copy()

EVALUATION_POSSIBILITIES = [
    "No danger",
    "Slightly dangerous",
    "Moderately dangerous",
    "Very dangerous",
]


# ============================================================
# 3. HUMAN-REFERENCE NORMALIZATION
# Reference aliases are retained only to protect against harmless
# historical spelling/capitalization variants in the Excel.
# They are NOT used to reinterpret constrained model predictions.
# ============================================================

GROUND_TRUTH_NATURAL_ALIASES = {
    "wildfire": "Wildfire",
    "wild fire": "Wildfire",
    "landslides/rockfalls": "Landslides/Rockfalls",
    "rockfall": "Landslides/Rockfalls",
    "rockfalls": "Landslides/Rockfalls",
    "landslide": "Landslides/Rockfalls",
    "landslides": "Landslides/Rockfalls",
    "flooding": "Flooding",
    "solar exposure": "Solar exposure",
    "volcanic activity": "Volcanic activity",
    "fog": "Fog",
    "snow/avalanche": "Snow/Avalanche",
    "wind": "Wind",
}

GROUND_TRUTH_SAFETY_ALIASES = {
    "steep path": "Steep path",
    "steep slopes on the side(s) of the paths": "Steep slopes on the side(s) of the paths",
    "steep slopes on the side(s) of the path": "Steep slopes on the side(s) of the paths",
    "steep slopes on the side(s)": "Steep slopes on the side(s) of the paths", 
    "steep slopes": "Steep slopes on the side(s) of the paths",
    "narrow path": "Narrow path",
    "damaged/irregular path": "Damaged/irregular path",
    "damaged path": "Damaged/irregular path",
    "irregular path": "Damaged/irregular path",
    "muddy/wet path": "Muddy/wet path",
    "mud": "Muddy/wet path",
    "wet path": "Muddy/wet path",
    "visitor overpresence": "Visitor overpresence",
    "visitor over presence": "Visitor overpresence",
    "broken/missing guardrails": "Broken/missing guardrails",
    "missing guardrails": "Broken/missing guardrails",
    "damaged guardrails": "Broken/missing guardrails",
    "faded/missing trail markers": "Faded/missing trail markers",
    "damaged stairs/boardwalks": "Damaged stairs/boardwalks",
    "damaged stairs": "Damaged stairs/boardwalks",
    "erosion": "Erosion",
    "overhanging branches": "Overhanging branches",
    "exposed stones/roots on the path": "Exposed stones/roots on the path",
    "exposed stones/roots": "Exposed stones/roots on the path",
    "exposed rocks": "Exposed stones/roots on the path",
    "dangerous flora/fauna": "Dangerous flora/fauna",
}


# ============================================================
# 4. PROMPT
# ============================================================

CHECKLIST_PROMPT = (
    f"Inspect the trail for natural hazards on this list: {', '.join(NATURAL_HAZARDS_CHECKLIST)}. "
    f"Inspect the trail for infrastructure problems on this list: {', '.join(INFRASTRUCTURE_PROBLEMS_CHECKLIST)}. "
    "Respond only with the name of the hazard or problem. Add nothing else. "
    "If multiple terms are identified per category, separate them using ';'. "
    "If none are identified, answer with 'None'. "
    "Under no circumstance leave cells blank. "
    "Identify which trail the photo belongs to. "
    "If the file has coordinates, use them for the location. If not identify visually. "
    f"Give your final evaluation of risk for visitors. Use only these terms: {', '.join(EVALUATION_POSSIBILITIES)}. "
    "Base your evaluation not only on the number of hazards/problems, but their severity as well."
)

SYSTEM_PROMPT = (
    "You are a trail safety inspector. "
    "Analyze images for natural hazards and trail infrastructure problems."
)


# ============================================================
# 5. STRUCTURED RESPONSE
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
# 6. SETUP AND GENERAL HELPERS
# ============================================================

client = OpenAI(api_key=OPENAI_API_KEY)
PREPROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)


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
        return 34853 in img.getexif()  # GPSInfo
    except Exception:
        return False


def atomic_save_json(data, path):
    temp_path = str(path) + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(temp_path, path)


def safe_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return None


# ============================================================
# 7. IMAGE PREPROCESSING
# ============================================================


def preprocess_image(source_path, destination_path, preserve_exif=True):
    with Image.open(source_path) as original:
        original_width, original_height = original.size
        original_has_exif = has_exif(original)
        original_has_gps = has_gps_exif(original)

        img = ImageOps.exif_transpose(original)
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
            Image.Resampling.LANCZOS,
        )

        processed_width, processed_height = img.size
        save_parameters = {"format": "JPEG", "quality": JPEG_QUALITY}
        if preserve_exif and exif_bytes:
            save_parameters["exif"] = exif_bytes

        img.save(destination_path, **save_parameters)

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
# 8. RAW MODEL INFERENCE
# No reference data are loaded here.
# ============================================================


def run_inference():
    wall_clock_start = time.perf_counter()

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
        "natural_hazards_checklist": NATURAL_HAZARDS_CHECKLIST,
        "infrastructure_problems_checklist": INFRASTRUCTURE_PROBLEMS_CHECKLIST,
        "danger_levels": EVALUATION_POSSIBILITIES,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "openai_package_version": safe_package_version("openai"),
        "pillow_version": safe_package_version("Pillow"),
        "pandas_version": safe_package_version("pandas"),
        "timing_definition": (
            "Successful API response latency observed by the client; preprocessing and other local work "
            "are excluded from per-image API latency; not pure model inference time."
        ),
    }

    records = []
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        filename
        for filename in os.listdir(IMAGE_FOLDER)
        if filename.lower().endswith(valid_extensions)
    )

    for index, filename in enumerate(image_files, start=1):
        original_path = IMAGE_FOLDER / filename
        processed_filename = Path(filename).stem + ".jpg"
        processed_path = PREPROCESSED_FOLDER / processed_filename
        print(f"[{index}/{len(image_files)}] Processing {filename}")

        try:
            preprocessing_info = preprocess_image(
                original_path,
                processed_path,
                preserve_exif=PRESERVE_EXIF,
            )
            base64_image = encode_image(processed_path)

            request_start = time.perf_counter()
            response = client.beta.chat.completions.parse(
                model=MODEL_ID,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CHECKLIST_PROMPT},
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
            request_end = time.perf_counter()
            api_latency = round(request_end - request_start, 3)

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
                "trail_name": analysis.trail_name,
                "natural_hazards": analysis.natural_hazards,
                "trail_problems": analysis.trail_problems,
                "evaluation": analysis.evaluation,
                "api_response_latency_seconds": api_latency,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            }
            records.append(record)
            print(f"  ✓ {filename}: {api_latency:.2f} seconds API latency")

        except Exception as exc:
            records.append({
                "status": "error",
                "original_filename": filename,
                "provider": PROVIDER,
                "requested_model": MODEL_ID,
                "prompt_type": PROMPT_TYPE,
                "run_id": RUN_ID,
                "error": str(exc),
            })
            print(f"  ✗ ERROR: {filename}: {exc}")

        atomic_save_json(
            {"experiment_metadata": experiment_metadata, "results": records},
            RAW_JSON_FILE,
        )

    successful_latencies = [
        r["api_response_latency_seconds"]
        for r in records
        if r.get("status") == "success" and r.get("api_response_latency_seconds") is not None
    ]
    wall_clock_seconds = round(time.perf_counter() - wall_clock_start, 2)
    total_processing = round(float(np.sum(successful_latencies)), 3) if successful_latencies else None

    experiment_metadata.update({
        "total_processing_seconds_excluding_waits": total_processing,
        "total_experiment_seconds": total_processing,
        "wall_clock_experiment_seconds": wall_clock_seconds,
        "successful_images": sum(r.get("status") == "success" for r in records),
        "failed_images": sum(r.get("status") != "success" for r in records),
        "mean_api_response_latency_seconds": round(float(np.mean(successful_latencies)), 3) if successful_latencies else None,
        "sd_api_response_latency_seconds": (
            round(float(np.std(successful_latencies, ddof=1)), 3) if len(successful_latencies) > 1 else None
        ),
    })
    atomic_save_json(
        {"experiment_metadata": experiment_metadata, "results": records},
        RAW_JSON_FILE,
    )
    print(f"\nRAW model output saved to:\n{RAW_JSON_FILE}")
    if total_processing is not None:
        print(f"Reported processing total (successful API calls only): {total_processing:.2f}s")
    print(f"Operational wall-clock duration: {wall_clock_seconds:.2f}s")


# ============================================================
# 9. TEXT PARSING AND STRICT CONSTRAINED-PREDICTION VALIDATION
# ============================================================


def clean_term(term):
    term = str(term).strip().lower()
    term = term.replace("–", "-").replace("—", "-")
    term = re.sub(r"^[\-\*\u2022]+\s*", "", term)
    term = re.sub(r"\s+", " ", term)
    return term.strip(" .,:\t\r\n")


def split_terms(text):
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return []
    text = str(text).strip()
    if not text:
        return []
    if clean_term(text) in {
        "none", "no", "no hazards", "no hazard", "no problems", "no problem", "n/a"
    }:
        return []
    return [clean_term(x) for x in re.split(r"[;\n]+", text) if clean_term(x)]


def normalize_ground_truth(raw_text, aliases, allowed_classes):
    normalized = set()
    for term in split_terms(raw_text):
        if term not in aliases:
            raise ValueError(
                f"Unknown human-reference term '{term}'. Correct the Excel or add it "
                "deliberately to the GROUND_TRUTH alias dictionary."
            )
        canonical = aliases[term]
        if canonical not in allowed_classes:
            raise ValueError(f"Reference term '{canonical}' is not a canonical class.")
        normalized.add(canonical)
    return normalized


def parse_constrained_prediction(raw_text, allowed_classes):
    """
    No semantic synonym mapping is performed.
    Only exact checklist labels are accepted, ignoring case/outer whitespace.
    Any out-of-taxonomy term is retained as INVALID::<raw term> so it is counted
    as a false positive and can be audited as a protocol violation.
    """
    canonical_by_clean = {clean_term(label): label for label in allowed_classes}
    prediction_set = set()
    invalid_terms = []

    for term in split_terms(raw_text):
        if term in canonical_by_clean:
            prediction_set.add(canonical_by_clean[term])
        else:
            invalid_label = f"INVALID::{term}"
            prediction_set.add(invalid_label)
            invalid_terms.append(term)

    return prediction_set, invalid_terms


# ============================================================
# 10. METRICS AND BOOTSTRAP
# ============================================================


def safe_divide(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def metrics_from_counts(tp, fp, fn):
    recall = safe_divide(tp, tp + fn)
    precision = safe_divide(tp, tp + fp)
    fdr = safe_divide(fp, tp + fp)
    rf = safe_divide(tp + fp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "Recall": recall, "Precision": precision, "FDR": fdr, "F1": f1, "RF": rf,
    }


def bootstrap_confidence_intervals(scored_rows, bootstrap_samples=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(scored_rows)
    metric_names = ["Recall", "Precision", "FDR", "F1", "RF"]
    if n == 0:
        return {m: {"CI95_lower": None, "CI95_upper": None} for m in metric_names}

    values = {m: [] for m in metric_names}
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        tp = sum(scored_rows[i]["TP"] for i in indices)
        fp = sum(scored_rows[i]["FP"] for i in indices)
        fn = sum(scored_rows[i]["FN"] for i in indices)
        metrics = metrics_from_counts(tp, fp, fn)
        for metric in metric_names:
            if metrics[metric] is not None:
                values[metric].append(metrics[metric])

    result = {}
    for metric, metric_values in values.items():
        if metric_values:
            result[metric] = {
                "CI95_lower": float(np.percentile(metric_values, 2.5)),
                "CI95_upper": float(np.percentile(metric_values, 97.5)),
            }
        else:
            result[metric] = {"CI95_lower": None, "CI95_upper": None}
    return result


# ============================================================
# 11. SCORE ONE HAZARD DOMAIN
# ============================================================


def score_domain(
    merged_dataframe,
    prediction_column,
    ground_truth_column,
    ground_truth_aliases,
    allowed_classes,
    domain_name,
):
    scored_rows = []
    invalid_output_rows = []

    for _, row in merged_dataframe.iterrows():
        filename = row["original_filename"]
        prediction_set, invalid_terms = parse_constrained_prediction(
            row[prediction_column], allowed_classes
        )
        reference_set = normalize_ground_truth(
            row[ground_truth_column], ground_truth_aliases, allowed_classes
        )

        tp_classes = prediction_set & reference_set
        fp_classes = prediction_set - reference_set
        fn_classes = reference_set - prediction_set

        scored_rows.append({
            "filename": filename,
            "domain": domain_name,
            "prediction_raw": row[prediction_column],
            "prediction_normalized": "; ".join(sorted(prediction_set)) if prediction_set else "None",
            "reference_raw": row[ground_truth_column],
            "reference_normalized": "; ".join(sorted(reference_set)) if reference_set else "None",
            "TP_classes": "; ".join(sorted(tp_classes)) if tp_classes else "None",
            "FP_classes": "; ".join(sorted(fp_classes)) if fp_classes else "None",
            "FN_classes": "; ".join(sorted(fn_classes)) if fn_classes else "None",
            "TP": len(tp_classes),
            "FP": len(fp_classes),
            "FN": len(fn_classes),
            "is_control": len(reference_set) == 0,
            "control_has_false_positive": len(reference_set) == 0 and len(prediction_set) > 0,
            "has_invalid_constrained_term": len(invalid_terms) > 0,
        })

        for term in invalid_terms:
            invalid_output_rows.append({
                "filename": filename,
                "domain": domain_name,
                "raw_term": term,
                "status": "Out-of-taxonomy constrained output",
            })

    total_tp = sum(row["TP"] for row in scored_rows)
    total_fp = sum(row["FP"] for row in scored_rows)
    total_fn = sum(row["FN"] for row in scored_rows)
    overall = metrics_from_counts(total_tp, total_fp, total_fn)
    overall["domain"] = domain_name
    overall["n_images"] = len(scored_rows)

    controls = [row for row in scored_rows if row["is_control"]]
    if controls:
        controls_with_fp = sum(row["control_has_false_positive"] for row in controls)
        total_control_fp = sum(row["FP"] for row in controls)
        overall.update({
            "control_images": len(controls),
            "controls_with_false_positive": controls_with_fp,
            "control_false_positive_image_rate": controls_with_fp / len(controls),
            "control_clear_rate": 1 - controls_with_fp / len(controls),
            "false_positive_classes_on_controls": total_control_fp,
            "mean_false_positive_classes_per_control": total_control_fp / len(controls),
        })
    else:
        overall.update({
            "control_images": 0,
            "controls_with_false_positive": 0,
            "control_false_positive_image_rate": None,
            "control_clear_rate": None,
            "false_positive_classes_on_controls": 0,
            "mean_false_positive_classes_per_control": None,
        })

    overall["invalid_constrained_terms"] = len(invalid_output_rows)

    ci = bootstrap_confidence_intervals(
        scored_rows, bootstrap_samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    for metric, interval in ci.items():
        overall[f"{metric}_CI95_lower"] = interval["CI95_lower"]
        overall[f"{metric}_CI95_upper"] = interval["CI95_upper"]

    class_rows = []
    for hazard_class in allowed_classes:
        class_tp = class_fp = class_fn = 0
        for _, source_row in merged_dataframe.iterrows():
            pred_set, _ = parse_constrained_prediction(
                source_row[prediction_column], allowed_classes
            )
            ref_set = normalize_ground_truth(
                source_row[ground_truth_column], ground_truth_aliases, allowed_classes
            )
            predicted = hazard_class in pred_set
            reference = hazard_class in ref_set
            if predicted and reference:
                class_tp += 1
            elif predicted and not reference:
                class_fp += 1
            elif not predicted and reference:
                class_fn += 1

        class_metrics = metrics_from_counts(class_tp, class_fp, class_fn)
        class_metrics["domain"] = domain_name
        class_metrics["class"] = hazard_class
        class_rows.append(class_metrics)

    return scored_rows, overall, class_rows, invalid_output_rows


# ============================================================
# 12. DANGER EVALUATION / CONFUSION MATRIX
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
# 13. LOAD RAW JSON + REFERENCE, SCORE, EXPORT
# ============================================================


def score_saved_raw_results():
    print("\nStarting scoring stage...")

    if not RAW_JSON_FILE.exists():
        raise FileNotFoundError(f"RAW JSON does not exist:\n{RAW_JSON_FILE}")

    with open(RAW_JSON_FILE, "r", encoding="utf-8") as f:
        raw_package = json.load(f)

    raw_results = raw_package["results"]
    successful_results = [r for r in raw_results if r.get("status") == "success"]
    if not successful_results:
        raise ValueError("No successful model results were found in the RAW JSON.")
    model_df = pd.DataFrame(successful_results)

    ground_truth_df = pd.read_excel(GROUND_TRUTH_EXCEL, sheet_name=GROUND_TRUTH_SHEET)
    required_columns = {
        GT_FILENAME_COL, GT_NATURAL_COL, GT_SAFETY_COL, GT_DANGER_COL
    }
    missing_columns = required_columns - set(ground_truth_df.columns)
    if missing_columns:
        raise ValueError(f"Ground-truth Excel is missing columns: {missing_columns}")

    model_df["_join_filename"] = (
        model_df["original_filename"].astype(str).str.strip()
        .apply(lambda x: Path(x).stem.lower())
    )
    ground_truth_df["_join_filename"] = (
        ground_truth_df[GT_FILENAME_COL].astype(str).str.strip()
        .apply(lambda x: Path(x).stem.lower())
    )

    if ground_truth_df["_join_filename"].duplicated(keep=False).any():
        dupes = ground_truth_df.loc[
            ground_truth_df["_join_filename"].duplicated(keep=False), GT_FILENAME_COL
        ].tolist()
        raise ValueError(f"Duplicate filenames in ground-truth Excel: {dupes}")

    if model_df["_join_filename"].duplicated(keep=False).any():
        dupes = model_df.loc[
            model_df["_join_filename"].duplicated(keep=False), "original_filename"
        ].tolist()
        raise ValueError(f"Duplicate filenames in model RAW JSON: {dupes}")

    ground_truth_for_merge = ground_truth_df[[
        "_join_filename", GT_NATURAL_COL, GT_SAFETY_COL, GT_DANGER_COL
    ]].rename(columns={
        GT_NATURAL_COL: "gt_natural_hazards",
        GT_SAFETY_COL: "gt_trail_problems",
        GT_DANGER_COL: "gt_danger",
    })

    merged = model_df.merge(
        ground_truth_for_merge,
        on="_join_filename",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        missing_names = merged.loc[unmatched, "original_filename"].tolist()
        raise ValueError(
            "These model images have no matching ground-truth row:\n"
            f"{missing_names}"
        )
    merged = merged.drop(columns=["_merge"])

    gt_without_model = sorted(
        set(ground_truth_df["_join_filename"]) - set(model_df["_join_filename"])
    )
    if gt_without_model:
        print("\nWARNING: reference rows with no successful model result:")
        for filename in gt_without_model:
            print(f"  - {filename}")

    natural_scored, natural_overall, natural_classwise, natural_invalid = score_domain(
        merged_dataframe=merged,
        prediction_column="natural_hazards",
        ground_truth_column="gt_natural_hazards",
        ground_truth_aliases=GROUND_TRUTH_NATURAL_ALIASES,
        allowed_classes=NATURAL_CLASSES,
        domain_name="Natural hazards",
    )

    safety_scored, safety_overall, safety_classwise, safety_invalid = score_domain(
        merged_dataframe=merged,
        prediction_column="trail_problems",
        ground_truth_column="gt_trail_problems",
        ground_truth_aliases=GROUND_TRUTH_SAFETY_ALIASES,
        allowed_classes=SAFETY_CLASSES,
        domain_name="Trail safety",
    )

    danger_comparison, danger_counts, danger_percent, danger_summary = (
        calculate_danger_results(merged)
    )

    scored_rows = natural_scored + safety_scored
    overall_statistics = [natural_overall, safety_overall]
    classwise_statistics = natural_classwise + safety_classwise
    invalid_terms = natural_invalid + safety_invalid

    scored_package = {
        "source_raw_json": str(RAW_JSON_FILE),
        "ground_truth_excel": str(GROUND_TRUTH_EXCEL),
        "scoring_method": "Micro-averaged pooled TP/FP/FN counts across images",
        "prediction_policy": (
            "Constrained outputs are scored only against exact checklist labels "
            "after case/whitespace normalization. No semantic synonym mapping is applied. "
            "Out-of-taxonomy terms are retained as invalid false-positive predictions."
        ),
        "scored_results": scored_rows,
        "danger_comparison": danger_comparison.to_dict(orient="records"),
    }
    atomic_save_json(scored_package, SCORED_JSON_FILE)

    statistics_package = {
        "overall_statistics": overall_statistics,
        "classwise_statistics": classwise_statistics,
        "invalid_constrained_terms": invalid_terms,
        "danger_summary": danger_summary.to_dict(orient="records"),
        "danger_confusion_counts": danger_counts.to_dict(),
        "danger_confusion_percent": danger_percent.to_dict(),
    }
    atomic_save_json(statistics_package, STATISTICS_JSON_FILE)

    scored_df = pd.DataFrame(scored_rows)
    overall_df = pd.DataFrame(overall_statistics)
    classwise_df = pd.DataFrame(classwise_statistics)
    invalid_df = pd.DataFrame(invalid_terms)
    if invalid_df.empty:
        invalid_df = pd.DataFrame(columns=["filename", "domain", "raw_term", "status"])

    metadata_df = pd.DataFrame([raw_package["experiment_metadata"]]).T.reset_index()
    metadata_df.columns = ["parameter", "value"]

    with pd.ExcelWriter(STATISTICS_EXCEL_FILE, engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="Overall_statistics", index=False)
        classwise_df.to_excel(writer, sheet_name="Classwise_statistics", index=False)
        scored_df.to_excel(writer, sheet_name="Per_image_scoring", index=False)
        invalid_df.to_excel(writer, sheet_name="Invalid_constrained_terms", index=False)

        # Raw per-image outputs include location, danger evaluation, EXIF/GPS status, and API latency.
        inference_df = model_df.drop(columns=["_join_filename"], errors="ignore").copy()
        inference_df.to_excel(writer, sheet_name="Inference_results", index=False)

        timing_values = pd.to_numeric(
            inference_df.get("api_response_latency_seconds", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        timing_summary_df = pd.DataFrame([{
            "n_successful_images": len(inference_df),
            "mean_api_response_latency_seconds": timing_values.mean() if len(timing_values) else None,
            "sd_api_response_latency_seconds": timing_values.std(ddof=1) if len(timing_values) > 1 else None,
            "median_api_response_latency_seconds": timing_values.median() if len(timing_values) else None,
            "total_processing_seconds_excluding_waits": raw_package["experiment_metadata"].get("total_processing_seconds_excluding_waits"),
            "total_experiment_seconds": raw_package["experiment_metadata"].get("total_experiment_seconds"),
            "wall_clock_experiment_seconds": raw_package["experiment_metadata"].get("wall_clock_experiment_seconds"),
        }])
        timing_summary_df.to_excel(writer, sheet_name="Timing_summary", index=False)
        danger_summary.to_excel(writer, sheet_name="Danger_summary", index=False)
        danger_comparison.to_excel(writer, sheet_name="Danger_comparison", index=False)
        danger_counts.to_excel(writer, sheet_name="Danger_confusion_counts")
        danger_percent.to_excel(writer, sheet_name="Danger_confusion_percent")
        metadata_df.to_excel(writer, sheet_name="Run_metadata", index=False)

    print("\nScoring complete.")
    print(f"\nScored JSON:\n{SCORED_JSON_FILE}")
    print(f"\nStatistics JSON:\n{STATISTICS_JSON_FILE}")
    print(f"\nStatistics Excel workbook:\n{STATISTICS_EXCEL_FILE}")
    print("\nOverall results:")
    print(overall_df.to_string(index=False))
    print("\nDanger confusion matrix (counts):")
    print(danger_counts.to_string())

    if invalid_terms:
        print(
            "\nNOTE: Some constrained outputs were outside the supplied taxonomy. "
            "They were retained as invalid false-positive predictions and are listed "
            "in the Invalid_constrained_terms sheet."
        )


# ============================================================
# 14. MAIN
# ============================================================

if __name__ == "__main__":
    if RUN_MODEL_INFERENCE:
        run_inference()
    if RUN_SCORING:
        score_saved_raw_results()
