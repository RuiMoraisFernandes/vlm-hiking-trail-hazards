import os, json, time, random
from google import genai
from google.genai import types
from pydantic import BaseModel

EVALUATION_POSSIBILITIES = [
    "Not dangerous",
    "Slightly dangerous",
    "Moderately dangerous",
    "Very dangerous"
]

# --- USER DEFINED --- 
API_KEY = "YOUR_API_KEY"
IMAGE_FOLDER = r"IMAGE_INPUT_FOLDER"
EXPORT_FOLDER = r"JSON_OUTPUT_FOLDER"
# --- USER DEFINED --- 

class AnalysisResult(BaseModel):
    trail_name: str
    natural_hazards: str
    trail_problems: str
    evaluation: str
    manual_timer_seconds: float 

client = genai.Client(api_key=API_KEY)
all_results = []
start_project_time = time.time()

checklist_prompt = (
    "Inspect the trail for natural hazards."
    "Inspect The trail for infrastructure problems."
    "Respond only with the name of the hazard or problem. Add nothing else."
    "If multiple terms are identified per category, separate them using ';'"
    "If none are identified, awswer with 'None'"
    "Under no circumstance leave cells blank."
    "Identify which trail the photo belongs to."
    "If the file has coordinates, use them for the location. If not identify visually."
    f"Give your final evaluation of risk for visitors. Use only these terms: {', '.join(EVALUATION_POSSIBILITIES)}."
    "Base your evaluation not only on the number of hazards/problems, but their severity as well." 
)

print(f"🚀 Starting analysis. Searching for images in {IMAGE_FOLDER}...")

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        
        # --- error 503 retries ---
        success = False
        retries = 0
        current_model = "gemini-3.1-flash-lite-preview"
        while not success and retries < 3:
            try:
                with open(file_path, "rb") as f:
                    img_bytes = f.read()

             
                t0 = time.time()

                image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite-preview",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AnalysisResult,
                        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level="high")
                ),
                contents=[image_part, checklist_prompt]
            )

           
                t1 = time.time()
                elapsed = round(t1 - t0, 2)

                if response.parsed:
                    final_record = {
                    "original_filename": filename,
                    "trail_name": response.parsed.trail_name,
                    "natural_hazards": response.parsed.natural_hazards,
                    "trail_problems": response.parsed.trail_problems,
                    "evaluation": response.parsed.evaluation,
                    "processing_time": elapsed
                }
                    all_results.append(final_record)
                    print(f"✅ {filename} processed in {elapsed}s | Saved time: {final_record['processing_time']}")
                    success = True
                
                time.sleep(4)

            except Exception as e:
                if "503" in str(e):
                    retries += 1
                    wait = (2 ** retries) * 5
                    print(f"⚠️ 503 Error. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"❌ Error: {e}")
                    break


with open(os.path.join(EXPORT_FOLDER, "FILE_NAME.json"), "w") as f: # --- USER DEFINED --- 
    json.dump(all_results, f, indent=4)

total_project_time = round(time.time() - start_project_time, 2)
print(f"Analysis complete in {total_project_time}s")