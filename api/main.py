


import io
import time
import logging
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np

from pathlib import Path
from PIL import Image
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ── Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants 
MODEL_PATH  = Path("models/efficientnet_inference.pth")
IMG_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
MAX_FILE_SIZE = 10 * 1024 * 1024          # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/bmp"}
MAX_BATCH     = 10

CLASS_DESCRIPTIONS = {
    "akiec": {
        "full_name"   : "Actinic Keratosis / Intraepithelial Carcinoma",
        "description" : "A rough, scaly patch caused by years of sun exposure.",
        "malignant"   : True,
    },
    "bcc": {
        "full_name"   : "Basal Cell Carcinoma",
        "description" : "The most common type of skin cancer, rarely spreads.",
        "malignant"   : True,
    },
    "bkl": {
        "full_name"   : "Benign Keratosis",
        "description" : "Non-cancerous skin growths including seborrheic keratoses.",
        "malignant"   : False,
    },
    "df": {
        "full_name"   : "Dermatofibroma",
        "description" : "A benign skin growth that often appears on the legs.",
        "malignant"   : False,
    },
    "mel": {
        "full_name"   : "Melanoma",
        "description" : "A serious form of skin cancer that develops from melanocytes.",
        "malignant"   : True,
    },
    "nv": {
        "full_name"   : "Melanocytic Nevi",
        "description" : "Common moles. Usually benign growths of melanocytes.",
        "malignant"   : False,
    },
    "vasc": {
        "full_name"   : "Vascular Lesions",
        "description" : "Includes angiomas, angiokeratomas, and pyogenic granulomas.",
        "malignant"   : False,
    },
}

# ── Global model state (populated at startup) 
app_state: dict = {}


# ── Model builder 
def build_efficientnet(num_classes: int) -> nn.Module:
    """Recreate the same architecture used in training."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ── Lifespan: load model once at startup 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights and metadata on startup, clean up on shutdown."""
    logger.info(" Starting HAM10000 Classifier API...")

    if not MODEL_PATH.exists():
        logger.error(f" Model file not found: {MODEL_PATH}")
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run Notebook 2 first to generate the model file."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f" Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Build model and load weights
    num_classes = checkpoint["num_classes"]
    model       = build_efficientnet(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((checkpoint["img_size"], checkpoint["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["imagenet_mean"],
            std=checkpoint["imagenet_std"]
        ),
    ])


    raw_idx_to_class = checkpoint["idx_to_class"]
    idx_to_class_normalized = {str(k): v for k, v in raw_idx_to_class.items()}

    app_state["model"]        = model
    app_state["device"]       = device
    app_state["transform"]    = transform
    app_state["idx_to_class"] = idx_to_class_normalized
    app_state["num_classes"]  = num_classes
    app_state["startup_time"] = time.time()

    logger.info(f" Model loaded — {num_classes} classes")
    logger.info(f"   Test Accuracy : {checkpoint.get('test_accuracy', 'N/A')}")
    logger.info(f"   Test Macro F1 : {checkpoint.get('test_macro_f1',  'N/A')}")
    logger.info(f"   Test ROC-AUC  : {checkpoint.get('test_roc_auc',   'N/A')}")

    yield  # ← app runs here

    logger.info(" Shutting down API...")
    app_state.clear()


# ── FastAPI app 
app = FastAPI(
    title       = "HAM10000 Skin Lesion Classifier",
    description = (
        "EfficientNet-B0 model trained on the HAM10000 dataset.\n\n"
        "Classifies dermoscopy images into 7 skin lesion categories.\n\n"
        "**This is a research tool — not a medical diagnostic device.**"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic response models 
class ClassProbability(BaseModel):
    class_key   : str
    full_name   : str
    probability : float
    percentage  : str


class PredictionResponse(BaseModel):
    filename          : str
    predicted_class   : str
    full_name         : str
    confidence        : float
    confidence_pct    : str
    is_malignant      : bool
    malignant_warning : str
    all_probabilities : List[ClassProbability]
    inference_time_ms : float


class BatchPredictionResponse(BaseModel):
    total_images : int
    results      : List[PredictionResponse]
    total_time_ms: float


class HealthResponse(BaseModel):
    status       : str
    model_loaded : bool
    device       : str
    num_classes  : int
    uptime_sec   : float


# ── Helpers 
def validate_image_file(file: UploadFile) -> None:
    """Raise HTTPException if file is not a valid image."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                f"Accepted types: {', '.join(ALLOWED_TYPES)}"
            ),
        )


async def read_image(file: UploadFile) -> Image.Image:
    """Read upload bytes and decode as PIL Image."""
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE // (1024*1024)} MB."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    return image


@torch.no_grad()
def run_inference(image: Image.Image) -> tuple:
    """
    Run model inference on a single PIL image.
    Returns (predicted_class_key, confidence, all_probs_dict).
    """
    model     = app_state["model"]
    device    = app_state["device"]
    transform = app_state["transform"]
    idx_to_class = app_state["idx_to_class"]

    # Preprocess
    tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    start   = time.perf_counter()
    logits  = model(tensor)
    elapsed = (time.perf_counter() - start) * 1000   # ms

    probs        = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx     = int(np.argmax(probs))
    pred_key     = idx_to_class[str(pred_idx)]
    confidence   = float(probs[pred_idx])

    all_probs = {
        idx_to_class[str(i)]: float(probs[i])
        for i in range(len(probs))
    }

    return pred_key, confidence, all_probs, elapsed


def build_prediction_response(
    filename : str,
    pred_key : str,
    confidence: float,
    all_probs : dict,
    elapsed_ms: float,
) -> PredictionResponse:
    """Assemble the full PredictionResponse object."""
    cls_info    = CLASS_DESCRIPTIONS[pred_key]
    is_malignant = cls_info["malignant"]

    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

    return PredictionResponse(
        filename        = filename,
        predicted_class = pred_key,
        full_name       = cls_info["full_name"],
        confidence      = round(confidence, 6),
        confidence_pct  = f"{confidence * 100:.2f}%",
        is_malignant    = is_malignant,
        malignant_warning = (
            " Potentially malignant lesion detected. Please consult a dermatologist."
            if is_malignant else
            " Classified as benign. Regular monitoring is still recommended."
        ),
        all_probabilities=[
            ClassProbability(
                class_key   = k,
                full_name   = CLASS_DESCRIPTIONS[k]["full_name"],
                probability = round(v, 6),
                percentage  = f"{v * 100:.2f}%",
            )
            for k, v in sorted_probs
        ],
        inference_time_ms=round(elapsed_ms, 2),
    )


# ── Routes 
@app.get("/", tags=["General"])
async def root():
    return {
        "message"    : "HAM10000 Skin Lesion Classifier API",
        "version"    : "1.0.0",
        "docs"       : "/docs",
        "health"     : "/health",
        "endpoints"  : {
            "GET  /health"         : "Health check",
            "GET  /classes"        : "List all 7 classes",
            "POST /predict"        : "Single image prediction",
            "POST /predict/batch"  : "Batch prediction (up to 10 images)",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Check if the API and model are healthy."""
    model_loaded = "model" in app_state
    uptime       = time.time() - app_state.get("startup_time", time.time())
    device_str   = str(app_state.get("device", "unknown"))

    return HealthResponse(
        status       = "ok" if model_loaded else "degraded",
        model_loaded = model_loaded,
        device       = device_str,
        num_classes  = app_state.get("num_classes", 0),
        uptime_sec   = round(uptime, 2),
    )


@app.get("/classes", tags=["General"])
async def get_classes():
    """Return all 7 skin lesion classes with descriptions."""
    return {
        "num_classes": len(CLASS_DESCRIPTIONS),
        "classes"    : [
            {
                "key"        : key,
                "full_name"  : info["full_name"],
                "description": info["description"],
                "malignant"  : info["malignant"],
            }
            for key, info in CLASS_DESCRIPTIONS.items()
        ],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Dermoscopy image (JPG/PNG)")):
    """
    Predict skin lesion class for a single image.

    - **file**: Upload a dermoscopy image (JPG or PNG, max 10 MB)

    Returns the predicted class, confidence score, and probabilities for all 7 classes.
    """
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

    validate_image_file(file)
    image = await read_image(file)

    logger.info(f" Predicting: {file.filename}  size={image.size}")

    try:
        pred_key, confidence, all_probs, elapsed_ms = run_inference(image)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    response = build_prediction_response(
        filename   = file.filename or "uploaded_image",
        pred_key   = pred_key,
        confidence = confidence,
        all_probs  = all_probs,
        elapsed_ms = elapsed_ms,
    )

    logger.info(
        f"   → {pred_key} ({confidence*100:.1f}%)  "
        f"malignant={response.is_malignant}  "
        f"time={elapsed_ms:.1f}ms"
    )
    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(..., description=f"Up to {MAX_BATCH} dermoscopy images")
):
    """
    Predict skin lesion class for multiple images at once.

    - **files**: Upload up to 10 dermoscopy images
    - Returns a list of predictions in the same order as input files
    """
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size is {MAX_BATCH}."
        )

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    logger.info(f" Batch prediction: {len(files)} images")

    batch_start = time.perf_counter()
    results     = []

    for file in files:
        validate_image_file(file)
        image = await read_image(file)

        try:
            pred_key, confidence, all_probs, elapsed_ms = run_inference(image)
            response = build_prediction_response(
                filename   = file.filename or "uploaded_image",
                pred_key   = pred_key,
                confidence = confidence,
                all_probs  = all_probs,
                elapsed_ms = elapsed_ms,
            )
            results.append(response)
        except Exception as e:
            logger.error(f"Inference error on {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed for {file.filename}: {str(e)}"
            )

    total_ms = (time.perf_counter() - batch_start) * 1000
    logger.info(f"   Batch done — total time: {total_ms:.1f}ms")

    return BatchPredictionResponse(
        total_images  = len(files),
        results       = results,
        total_time_ms = round(total_ms, 2),
    )
