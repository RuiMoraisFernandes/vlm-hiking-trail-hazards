import os, json, time, base64
from PIL import Image
import io
from groq import Groq

# --- 1. Config. ---

# --- USER DEFINED --- 
API_KEY = "YOUR_API_KEY"
IMAGE_FOLDER = r"IMAGE_INPIT_FOLDER"
EXPORT_FOLDER = r"JSON_OUTPUT_FOLDER"
# --- USER DEFINED --- 

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Lists
NATURAL_HAZARDS_CHECKLIST = [
    "Wild fire",
    "Landslides/rockfalls",
    "Flooding",
    "Solar exposure",
    "Volcanic activity",
    "Fog",
    "Snow/Avalanche",
    "Wind"
]
INFRASTRUCTURE_PROBLEMS_CHECKLIST = [
    "Steep path",
    "Steep slopes on the side(s) of the path",
    "Narrow path",
    "Damaged/irregular path",
    "Muddy/wet path",
    "Visitor over presence",
    "Broken or missing guardrails",
    "Faded or missing trail markers",
    "Damaged stairs/boardwalks",
    "Erosion",
    "Overhanging branches",
    "Exposed stones/roots on the path",
    "Dangerous flora/fauna"
]
EVALUATION_POSSIBILITIES = [
    "Not dangerous",
    "Slightly dangerous",
    "Moderately dangerous",
    "Very dangerous"
]

client = Groq(API_KEY)
start_project_time = time.time()

def encode_image(image_path):
    """Resizes image to under 4MB limit and returns base64 string"""
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        img.thumbnail((1500, 1500))
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85) 
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

all_results = []
checklist_prompt = (
    f"Inspect the trail for these NATURAL HAZARDS: {', '.join(NATURAL_HAZARDS_CHECKLIST)}. "
    f"Inspect INFRASTRUCTURE PROBLEMS: {', '.join(INFRASTRUCTURE_PROBLEMS_CHECKLIST)}. "
    "Respond only with the name of the hazard or problem. Add nothing else."
    "If multiple terms are identified per category, separate them using ';'"
    "If none are identified, awswer with 'None'"
    "Under no circumstance leave cells blank."
    "Identify which trail the photo belongs to."
    "If the file has coordinates, use them for the location. If not identify visually."
    f"Give your final evaluation of risk for visitors. Use only these terms: {', '.join(EVALUATION_POSSIBILITIES)}."
    "Base your evaluation not only on the number of hazards/problems, but their severity as well." 
    # Necessary instruction in-prompt for the saving of the results in JSON format 
    "Follow this exact JSON structure: {"
    "\"trail_location\": \"Trail name or 'Unknown'\", "
    "\"natural_hazards\": \"List hazards or 'None'\", "
    "\"trail_problems\": \"List problems or 'None'\", "
    "\"location\": \"Trail name or coordinates\", "
    "\"evaluation\": \"Risk level\""
    "\"eprocessing_time\": \"elapsed\""
    "}." 
)

files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in files:
    file_path = os.path.join(IMAGE_FOLDER, filename)
    t0 = time.time()
    
    try:
        base64_image = encode_image(file_path)
        
        # Llama Call
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": checklist_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )

        elapsed = round(time.time() - t0, 2)
        data = json.loads(completion.choices[0].message.content)
        
        all_results.append({
            "original_filename": filename,
            "trail_location": data.get("trail_location", ""),
            "natural_hazards": data.get("natural_hazards", ""),
            "trail_problems": data.get("trail_problems", ""),
            "evaluation": data.get("evaluation", ""),
            "processing_time": elapsed
        })
        
        print(f"⚡ {filename} analyzed in {elapsed}s")
        
        
        time.sleep(2)

    except Exception as e:
        print(f"❌ Error on {filename}: {e}")
        if "429" in str(e):
            print("Rate limit hit. Sleeping for 30s...")
            time.sleep(30)

with open(os.path.join(EXPORT_FOLDER, "FILE_NAME.json"), "w") as f: # --- USER DEFINED --- 
    json.dump(all_results, f, indent=4)

    total_project_time = round(time.time() - start_project_time, 2)
print(f"Analysis complete in {total_project_time}s")
