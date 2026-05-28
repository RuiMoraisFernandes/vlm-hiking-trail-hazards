import os, json, time, base64, random, time, io
from mistralai.client import Mistral
from pydantic import BaseModel
from PIL import Image

#Lists
EVALUATION_POSSIBILITIES = [
    "Not dangerous",
    "Slightly dangerous",
    "Moderately dangerous",
    "Very dangerous"
]

# --- 1. Config. ---

# --- USER DEFINED --- 
API_KEY = "YOUR_API_KEY"
IMAGE_FOLDER = r"IMAGE_INPUT_FOLDER"
EXPORT_FOLDER = r"JSON_OUTPUT_FOLDER"
# --- USER DEFINED --- 

def clean_val(val):
    if val is None:
        return ""
    if isinstance(val, dict):
        return ", ".join([f"{k}: {v}" for k, v in val.items()])
    if isinstance(val, list):
        return "; ".join([str(i) for i in val])
    return str(val)

client = Mistral(api_key=API_KEY)
model = "mistral-large-latest"
start_project_time = time.time()

def encode_image(image_path):
    """Resizes and compresses image to stay under token limits"""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


all_results = []
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
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
        
    file_path = os.path.join(IMAGE_FOLDER, filename)
    success = False
    

    while not success:
        try:
            t0 = time.time()
            image_data = encode_image(file_path)
            
            chat_response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": checklist_prompt},
                    {"type": "image_url", "image_url": image_data}
                ]}],
                response_format={"type": "json_object"}
            )

            raw_content = chat_response.choices[0].message.content
            data = json.loads(raw_content)
            
            all_results.append({
                "original_filename": filename,
                "trail_name": clean_val(data.get("trail_name", "Unknown")),
                "natural_hazards": clean_val(data.get("natural_hazards", "")),
                "trail_problems": clean_val(data.get("trail_problems", "")),
                "evaluation": clean_val(data.get("evaluation", "")),
                "processing_time": round(time.time() - t0, 2)
            })
                
            print(f"✅ {filename} processed successfully.")
            success = True
            
            print("⏳ Mandantory 60s cooldown to prevent debt cycle...")
            time.sleep(20)

        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ Rate limit hit for {filename}. Standing down for 120s...")
                time.sleep(20)
            else:
                print(f"❌ Fatal Error on {filename}: {e}")
                break 

with open(os.path.join(EXPORT_FOLDER, "miFILE_NAME.json"), "w") as f: # --- USER DEFINED ---
    json.dump(all_results, f, indent=4)

total_project_time = round(time.time() - start_project_time, 2)
print(f"Analysis complete in {total_project_time}s")