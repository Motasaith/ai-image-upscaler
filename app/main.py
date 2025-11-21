from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import cv2
import numpy as np
import os
import uuid
from app.restoration import ImageRestorer

# Create output folder
os.makedirs("processed_images", exist_ok=True)

app = FastAPI(title="AI Image Restoration API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/processed", StaticFiles(directory="processed_images"), name="processed")

restorer = None
# Global Job Store (In-memory database)
# Format: { "job_id": { "status": "processing" | "completed" | "failed", "results": [] } }
job_store = {}

@app.on_event("startup")
async def startup_event():
    global restorer
    try:
        print("--- Loading AI Models... ---")
        restorer = ImageRestorer()
        print("--- Models Loaded Successfully ---")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

# --- THE BACKGROUND WORKER ---
def process_images_background(job_id: str, file_data_list: List[bytes], filenames: List[str], face_enhance: bool):
    """This runs in the background after the API responds"""
    results = []
    try:
        for i, file_bytes in enumerate(file_data_list):
            try:
                # Decode
                nparr = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process
                # Note: restorer is global
                processed_img = restorer.process_image(img, face_enhance=face_enhance)
                
                if processed_img is not None:
                    # Save
                    out_name = f"{uuid.uuid4()}.jpg"
                    out_path = os.path.join("processed_images", out_name)
                    cv2.imwrite(out_path, processed_img)
                    
                    results.append({
                        "original_filename": filenames[i],
                        "url": f"/processed/{out_name}"
                    })
            except Exception as inner_e:
                print(f"Error processing file {filenames[i]}: {inner_e}")

        # Update Job Store
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["results"] = results
        print(f"Job {job_id} finished.")

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        job_store[job_id]["status"] = "failed"

# --- ENDPOINTS ---

@app.post("/enhance")
async def submit_job(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...), 
    face_enhance: bool = True
):
    if restorer is None:
        raise HTTPException(status_code=500, detail="AI Models not loaded.")

    # 1. Read all files into memory immediately
    file_data_list = []
    filenames = []
    for file in files:
        if file.content_type.startswith('image/'):
            content = await file.read()
            file_data_list.append(content)
            filenames.append(file.filename)

    # 2. Create a Job ID
    job_id = str(uuid.uuid4())
    job_store[job_id] = { "status": "processing", "results": [] }

    # 3. Start Background Task (This happens AFTER we return the ID)
    background_tasks.add_task(process_images_background, job_id, file_data_list, filenames, face_enhance)

    # 4. Return ID immediately (Fast response!)
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Processing started in background"
    }

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_store[job_id]