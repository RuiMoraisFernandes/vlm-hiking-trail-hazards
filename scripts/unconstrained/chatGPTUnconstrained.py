import os, json, time, base64
from openai import OpenAI
from pydantic import BaseModel

#Lists
EVALUATION_POSSIBILITIES = [
    "Not dangerous",
    "Slightly dangerous",
    "Moderately dangerous",
    "Very dangerous"
]

# --- 1. Config. ---

# --- USER DEFINED --- 
OPENAI_API_KEY = "YOUR_API_KEY"
IMAGE_FOLDER = r"INPUT_IMAGE_FOLDER"
EXPORT_FOLDER = r"OUTPUT_IMAGE_FOLDER"
# --- USER DEFINED --- 

client = OpenAI(api_key=OPENAI_API_KEY)
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

class AnalysisResult(BaseModel):
    trail_name: str
    natural_hazards: str
    trail_problems: str
    evaluation: str
    manual_timer_seconds: float 

# Helper function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

all_results = []

# --- 2. The loop ---
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        t_start = time.time()
        
        try:
            base64_image = encode_image(file_path)

            
            response = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are a trail safety inspector. Analyze images for hazards and trail infrastructure problems."},
                    {"role": "user", "content": [
                        {"type": "text", "text": checklist_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                response_format=AnalysisResult,
            )

            actual_seconds = round(time.time() - t_start, 2)
            
           
            analysis = response.choices[0].message.parsed
            
            final_record = {
                "original_filename": filename,
                "trail_name": analysis.trail_name,
                "natural_hazards": analysis.natural_hazards,
                "trail_problems": analysis.trail_problems,
                "evaluation": analysis.evaluation,
                "processing_time": actual_seconds
            }
            
            all_results.append(final_record)
            print(f"✅ ChatGPT: {filename} | {actual_seconds}s")

        except Exception as e:
            print(f"❌ Error on {filename}: {e}")


with open(os.path.join(EXPORT_FOLDER, "FILE_NAME.json"), "w") as f: # --- USER DEFINED --- 
    json.dump(all_results, f, indent=4)

total_project_time = round(time.time() - start_project_time, 2)
print(f"Analysis complete in {total_project_time}s")