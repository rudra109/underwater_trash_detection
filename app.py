from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLO
from datetime import datetime
import shutil, uuid, os

# -- APP INIT 
app = FastAPI(title="Underwater Pollution Intelligence API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best (1).pt") #model yolo v8

model = YOLO(MODEL_PATH)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)



#  main api
@app.post("/detect/")
async def detect_trash(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    depth: float = Form(None)   # optional depth
):
    # Timestamp (UTC)
    timestamp = datetime.utcnow().isoformat()

    # Save uploaded image
    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)













    # Run YOLO detection (save crops for image_clip)
    results = model.predict(
        filepath,
        conf=0.25,
        save=True,
        save_crop=True
    )

    records = []

    for r in results:
        if r.boxes:
            for i, box in enumerate(r.boxes):
                class_name = r.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                # YOLO crop path (relative)
                crop_path = None
                if r.save_dir:
                    crop_path = os.path.join(
                        r.save_dir,
                        "crops",
                        class_name,
                        f"{i}.jpg"
                    )

                records.append({
                    "datetime": timestamp,
                    "lat": latitude,
                    "lon": longitude,
                    "depth": depth,
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "image_clip": crop_path
                })

    return {
        "status": "success",
        "detections": records
    }
