"""REST API: upload a floor-plan image or PDF, get structured building data back as JSON.

    MODEL_PATH=checkpoints/best.pth uvicorn app.api:app --reload
    curl -F "file=@plan.png" "http://127.0.0.1:8000/analyse?mc_samples=8"
"""
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fpintel.inference import Predictor  # noqa: E402
from fpintel.render import analyse, load_drawing  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/best.pth")
CALIBRATION_PATH = os.environ.get("CALIBRATION_PATH", str(Path(MODEL_PATH).parent / "calibration.json"))
MAX_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Floor-plan intelligence prototype", version="0.1.0")
_predictor = None


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor(MODEL_PATH, CALIBRATION_PATH)
    return _predictor


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/analyse")
async def analyse_drawing(file: UploadFile = File(...),
                          mc_samples: int = Query(8, ge=1, le=32),
                          review_threshold: float = Query(0.8, ge=0.0, le=1.0)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")
    try:
        image = load_drawing(data, file.filename or "")
    except Exception as e:
        raise HTTPException(400, f"Could not read drawing: {e}")
    result, _ = analyse(predictor(), image, mc_samples, review_threshold)
    result["source_file"] = file.filename
    return result
