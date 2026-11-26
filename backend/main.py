from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import io
import logging
import cv2
import base64

# ==================================
# PROMETHEUS-X CONFIGURATION
# ==================================

app = FastAPI(
    title="PROMETHEUS-X",
    description="AI Vehicle Inspection System",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PROMETHEUS-X")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==================================
# MODEL LOADING
# ==================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

model = None
model_source = "none"


def load_model():
    global model, model_source

    # Try custom trained model
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100000:
        try:
            model = YOLO(str(MODEL_PATH))
            model_source = "custom"
            logger.info("Custom AI model loaded successfully")
            return
        except Exception as e:
            logger.error(f"Custom model failed: {e}")

    # Fallback model
    try:
        logger.info("Loading YOLOv8 pretrained model...")
        model = YOLO("yolov8n.pt")
        model_source = "pretrained"
        logger.info("Pretrained model loaded successfully")

    except Exception as e:
        model = None
        model_source = "none"
        logger.error(f"Model loading failed: {e}")


load_model()


# ==================================
# LOCATION DETECTION
# ==================================

def find_location(x1, y1, x2, y2, width, height):

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if cx < width / 3:
        horizontal = "left"
    elif cx > 2 * width / 3:
        horizontal = "right"
    else:
        horizontal = "center"

    if cy < height / 3:
        vertical = "upper"
    elif cy > 2 * height / 3:
        vertical = "lower"
    else:
        vertical = "middle"

    return f"{vertical}-{horizontal}"


# ==================================
# HOME API
# ==================================

@app.get("/")
def home():

    return {
        "project": "PROMETHEUS-X",
        "status": "Backend running",
        "model_loaded": model is not None,
        "model_source": model_source
    }


# ==================================
# HEALTH API
# ==================================

@app.get("/health")
def health():

    return {
        "status": "healthy" if model else "model_unavailable",
        "model_loaded": model is not None,
        "model_source": model_source
    }


# ==================================
# IMAGE PREDICTION API
# ==================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="AI model unavailable"
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please upload an image"
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty image"
        )

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB"
        )

    try:
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image.load()

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    width, height = image.size

    # ==================================
    # YOLO AI PREDICTION
    # ==================================

    try:
        results = model.predict(
            source=image,
            conf=0.25,
            verbose=True,
        )

        result = results[0]

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(
            status_code=500,
            detail="AI prediction failed"
        )

    detections = []
    confidence_values = []
    locations = []

    # ==================================
    # PROCESS DETECTIONS
    # ==================================

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().tolist()
            )

            confidence = float(
                box.conf[0].cpu().item()
            )

            class_id = int(
                box.cls[0].cpu().item()
            )

            label = str(model.names[class_id])

            location = find_location(
                x1, y1, x2, y2,
                width, height
            )

            confidence_values.append(confidence * 100)
            locations.append(location)

            detections.append({

                "label": label,

                "defect": label,

                "confidence": round(
                    confidence * 100, 2
                ),

                "location": location,

                "x": round(x1 / width, 4),
                "y": round(y1 / height, 4),

                "width": round(
                    (x2 - x1) / width, 4
                ),

                "height": round(
                    (y2 - y1) / height, 4
                ),

                "box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

    # ==================================
    # ANALYSIS RESULT
    # ==================================

    total = len(detections)

    max_confidence = (
        round(max(confidence_values), 2)
        if confidence_values else 0
    )

    anomaly = (
        ", ".join(set(
            item["label"] for item in detections
        ))
        if total > 0 else "No objects detected"
    )

    severity = (
        "Review Required" if total > 0
        else "Not Detected"
    )

    recommendation = (
        "Please review the detected objects. "
        "A vehicle-damage-specific trained model "
        "is required to confirm actual damage."
        if total > 0
        else "No objects detected. Try another image."
    )

    location = (
        ", ".join(set(locations))
        if locations else "Unknown"
    )

    # ==================================
    # ANNOTATED IMAGE
    # ==================================

    try:
        annotated = result.plot()

        success, buffer = cv2.imencode(
            ".jpg", annotated
        )

        if not success:
            raise ValueError("Image encoding failed")

        encoded_image = base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

    except Exception as e:
        logger.exception("Image annotation failed")
        raise HTTPException(
            status_code=500,
            detail="Annotated image generation failed"
        )

    # ==================================
    # FINAL RESPONSE
    # ==================================

    return {

        "success": True,

        "project": "PROMETHEUS-X",

        "model_source": model_source,

        "anomaly": anomaly,

        "severity": severity,

        "confidence": max_confidence,

        "location": location,

        "recommendation": recommendation,

        "detections": detections,

        "is_defect": total > 0,

        "defect_found": total > 0,

        "total_detections": total,

        "annotated_image": encoded_image
    }


# ==================================
# RUN SERVER
# ==================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        reload=False
    )