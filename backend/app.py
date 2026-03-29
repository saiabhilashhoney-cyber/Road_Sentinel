import os
import pathlib
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix for loading models trained on Posix systems on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import cv2
from paddleocr import PaddleOCR
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = BASE_DIR / "road_sentinel.db"

for folder in [UPLOAD_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Road Sentinel API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


helmet_model = YOLO(str(MODEL_DIR / "helmet_model.pt"))
plate_model = YOLO(str(MODEL_DIR / "plate_model.pt"))
# Cache PaddleOCR model
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


class ProcessResult(BaseModel):
    record_id: int
    case_id: str
    media_type: str
    original_url: str
    annotated_url: str
    plate_crop_url: Optional[str]
    plate_number: str
    confidence: float
    helmet_conf: float
    plate_conf: float
    helmet_status: str
    violation_status: str
    fine_status: str
    repeat_offender: bool
    risk_level: str
    seen_count: int
    created_at: str


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            helmet_status TEXT,
            violation_status TEXT,
            fine_status TEXT,
            confidence REAL,
            repeat_offender INTEGER,
            media_type TEXT,
            original_url TEXT,
            annotated_url TEXT,
            plate_crop_url TEXT,
            created_at TEXT
        )
        """
    )
    try:
        cursor.execute("ALTER TABLE records ADD COLUMN case_id TEXT")
        cursor.execute("ALTER TABLE records ADD COLUMN helmet_conf REAL")
        cursor.execute("ALTER TABLE records ADD COLUMN plate_conf REAL")
        cursor.execute("ALTER TABLE records ADD COLUMN risk_level TEXT")
        cursor.execute("ALTER TABLE records ADD COLUMN seen_count INTEGER")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def is_video(filename: str) -> bool:
    ext = filename.lower().split(".")[-1]
    return ext in {"mp4", "avi", "mov", "mkv", "webm"}


def is_image(filename: str) -> bool:
    ext = filename.lower().split(".")[-1]
    return ext in {"jpg", "jpeg", "png", "bmp", "webp"}


def class_label_map(names: Dict[int, str], class_id: int) -> str:
    try:
        return str(names.get(class_id, f"class_{class_id}"))
    except Exception:
        return f"class_{class_id}"


def normalize_helmet_label(raw: str) -> str:
    lower = raw.lower().replace("-", "_").strip()
    if "no_helmet" in lower or "without" in lower or "nohelmet" in lower:
        return "No Helmet"
    if "helmet" in lower:
        return "Helmet"
    return raw.title()


def detect_on_frame(frame: np.ndarray, target_conf: float = 0.45, enable_helmet: bool = True) -> Tuple[np.ndarray, Dict]:
    annotated = frame.copy()
    helmet_detections = []
    plate_detections = []

    if enable_helmet:
        helmet_results = helmet_model(frame, verbose=False, conf=target_conf)[0]
        for box in helmet_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw = class_label_map(helmet_model.names, cls_id)
            label = normalize_helmet_label(raw)

            color = (0, 255, 0) if label == "Helmet" else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            helmet_detections.append({"label": label, "conf": conf, "bbox": [x1, y1, x2, y2]})

    plate_results = plate_model(frame, verbose=False, conf=target_conf)[0]
    for box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            annotated,
            f"Number Plate {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )
        plate_detections.append({"conf": conf, "bbox": [x1, y1, x2, y2]})

    return annotated, {"helmet": helmet_detections, "plate": plate_detections}


def crop_best_plate(frame: np.ndarray, plate_detections: List[Dict]) -> Optional[np.ndarray]:
    if not plate_detections:
        return None
    best = max(plate_detections, key=lambda d: d["conf"])
    x1, y1, x2, y2 = best["bbox"]
    h, w = frame.shape[:2]
    # Add 6% padding so letters at the edge are never clipped
    pad_x = max(8, int((x2 - x1) * 0.06))
    pad_y = max(6, int((y2 - y1) * 0.06))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


import re as _re


def deskew_plate(img: np.ndarray) -> np.ndarray:
    """Detect skew angle via minAreaRect and rotate to be level."""
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, binary = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 20:
        return img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    # Only correct meaningful, non-extreme skew
    if abs(angle) < 0.5 or abs(angle) > 30:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _preprocess_for_ocr(img: np.ndarray, variant: int = 0) -> np.ndarray:
    """Enhanced preprocessing: grayscale, upscale, contrast adjustment, and binarization."""
    if img is None or img.size == 0:
        return img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    
    # Increase resolution for better OCR accuracy (cubic interpolation for quality)
    h, w = gray.shape[:2]
    scale = 2.5
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    if variant == 0:
        # High Accuracy: CLAHE + Otsu Binarization
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif variant == 1:
        # Fallback 1: High Contrast + Sharpening
        gray = cv2.equalizeHist(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
    elif variant == 2:
        # Fallback 2: Adaptive Thresholding for uneven lighting
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
    return gray


def ocr_plate(plate_img: Optional[np.ndarray]) -> str:
    """High-accuracy multi-pass OCR using PaddleOCR with preprocessing fallbacks."""
    if plate_img is None or plate_img.size == 0:
        return "Plate Not Detected"

    best_text = ""
    best_conf = 0.0

    # Try different preprocessing variants (Multi-pass fallback system)
    for variant in range(3):
        processed = _preprocess_for_ocr(plate_img, variant=variant)
        
        try:
            # PaddleOCR returns a list: [ [ [box, [text, conf]], ... ] ]
            results = ocr_reader.ocr(processed, cls=True)
            
            if results and results[0]:
                current_text = ""
                total_conf = 0.0
                count = 0
                
                for line in results[0]:
                    text, conf = line[1]
                    current_text += text
                    total_conf += conf
                    count += 1
                
                if count > 0:
                    avg_conf = total_conf / count
                    # Clean the current text (keep alphanumeric, uppercase)
                    clean_current = _re.sub(r'[^A-Z0-9]', '', current_text.upper())
                    
                    if avg_conf > best_conf and len(clean_current) >= 3:
                        best_conf = avg_conf
                        best_text = clean_current
                        
                    # Early exit if very high confidence to save processing time
                    if best_conf > 0.90:
                        break
        except Exception:
            continue

    if not best_text or best_conf < 0.2:
        return "UNREADABLE"

    return best_text


def summarize_violation(helmet_detections: List[Dict]) -> Tuple[str, str, str, float]:
    if not helmet_detections:
        return "Unknown", "Safe", "No Fine", 0.0
    
    # Priority: If any detection indicates a missing helmet, the overall frame is a violation
    # The label is normalized in detect_on_frame, so "Helmet" means compliant.
    violations = [d for d in helmet_detections if d["label"] != "Helmet"]
    
    if violations:
        best_violation = max(violations, key=lambda d: d["conf"])
        return "NO", "Violation", "Unpaid - ₹500", float(best_violation["conf"])
    else:
        best_safe = max(helmet_detections, key=lambda d: d["conf"])
        return "YES", "Safe", "No Fine", float(best_safe["conf"])


def insert_record(
    plate_number: str,
    helmet_status: str,
    violation_status: str,
    fine_status: str,
    confidence: float,
    helmet_conf: float,
    plate_conf: float,
    media_type: str,
    original_url: str,
    annotated_url: str,
    plate_crop_url: Optional[str],
) -> Dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    if plate_number not in {"UNREADABLE", "NOT_DETECTED", "Plate Not Detected"}:
        cur.execute("SELECT COUNT(*) FROM records WHERE plate_number = ?", (plate_number,))
        past_count = cur.fetchone()[0]
        seen_count = past_count + 1
        repeat_offender = 1 if past_count > 0 else 0
    else:
        seen_count = 0
        repeat_offender = 0

    if seen_count == 0 or seen_count == 1:
        seen_risk = "Low"
    elif seen_count in [2, 3]:
        seen_risk = "Medium"
    else:
        seen_risk = "High"

    # Risk level is primarily driven by helmet status — no helmet is always High risk
    if violation_status == "Violation":
        risk_level = "High"
    elif seen_risk == "Medium" or seen_risk == "High":
        risk_level = seen_risk  # repeat safe plate still gets medium/high for repeat tracking
    else:
        risk_level = "Low"

    now = datetime.now()
    created_at = now.isoformat()

    cur.execute(
        """
        INSERT INTO records (
            plate_number, helmet_status, violation_status, fine_status, confidence, repeat_offender,
            media_type, original_url, annotated_url, plate_crop_url, created_at,
            helmet_conf, plate_conf, risk_level, seen_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            plate_number, helmet_status, violation_status, fine_status, confidence, repeat_offender,
            media_type, original_url, annotated_url, plate_crop_url, created_at,
            helmet_conf, plate_conf, risk_level, seen_count
        ),
    )
    record_id = cur.lastrowid
    case_id = f"CASE-{now.year}-{int(record_id):04d}"
    
    cur.execute("UPDATE records SET case_id = ? WHERE id = ?", (case_id, record_id))
    conn.commit()
    conn.close()
    
    return {
        "record_id": int(record_id),
        "case_id": case_id,
        "seen_count": seen_count,
        "risk_level": risk_level,
        "created_at": created_at,
        "repeat_offender": bool(repeat_offender)
    }


def process_image(file_path: Path, output_id: str, target_conf: float, enable_helmet: bool) -> Dict:
    frame = cv2.imread(str(file_path))
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    annotated, det = detect_on_frame(frame, target_conf, enable_helmet)
    
    plate_img = crop_best_plate(frame, det["plate"])
    plate_number = ocr_plate(plate_img)
    
    plate_conf = 0.0
    if det["plate"]:
        plate_conf = max(det["plate"], key=lambda d: d["conf"])["conf"]
        
    helmet_status, violation_status, fine_status, helmet_conf = summarize_violation(det["helmet"])

    annotated_path = OUTPUT_DIR / f"{output_id}_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    plate_path = None
    if plate_img is not None:
        plate_path = OUTPUT_DIR / f"{output_id}_plate.jpg"
        cv2.imwrite(str(plate_path), plate_img)

    return {
        "media_type": "image",
        "annotated_path": annotated_path,
        "plate_path": plate_path,
        "plate_number": plate_number,
        "helmet_status": helmet_status,
        "violation_status": violation_status,
        "fine_status": fine_status,
        "confidence": max(helmet_conf, plate_conf),
        "helmet_conf": helmet_conf,
        "plate_conf": plate_conf,
    }


def process_video(file_path: Path, output_id: str, target_conf: float, enable_helmet: bool) -> Dict:
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    annotated_path = OUTPUT_DIR / f"{output_id}_annotated.mp4"
    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

    best_plate_img = None
    best_plate_conf = 0.0
    
    best_violation_det: Dict = {}
    best_safe_det: Dict = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, det = detect_on_frame(frame, target_conf, enable_helmet)
        writer.write(annotated)

        if det["helmet"]:
            for d in det["helmet"]:
                if d["label"] != "Helmet":
                    if not best_violation_det or d["conf"] > best_violation_det.get("conf", 0.0):
                        best_violation_det = d
                else:
                    if not best_safe_det or d["conf"] > best_safe_det.get("conf", 0.0):
                        best_safe_det = d

        if det["plate"]:
            candidate = max(det["plate"], key=lambda d: d["conf"])
            if candidate["conf"] > best_plate_conf:
                plate_img = crop_best_plate(frame, [candidate])
                if plate_img is not None:
                    best_plate_img = plate_img
                    best_plate_conf = candidate["conf"]

    cap.release()
    writer.release()

    final_helmet_det = []
    if best_violation_det:
        final_helmet_det.append(best_violation_det)
    elif best_safe_det:
        final_helmet_det.append(best_safe_det)

    plate_number = ocr_plate(best_plate_img)
    helmet_status, violation_status, fine_status, helmet_conf = summarize_violation(final_helmet_det)

    plate_path = None
    if best_plate_img is not None:
        plate_path = OUTPUT_DIR / f"{output_id}_plate.jpg"
        cv2.imwrite(str(plate_path), best_plate_img)

    return {
        "media_type": "video",
        "annotated_path": annotated_path,
        "plate_path": plate_path,
        "plate_number": plate_number,
        "helmet_status": helmet_status,
        "violation_status": violation_status,
        "fine_status": fine_status,
        "confidence": max(helmet_conf, best_plate_conf),
        "helmet_conf": helmet_conf,
        "plate_conf": best_plate_conf,
    }

from fastapi import Form

@app.post("/api/process", response_model=ProcessResult)
async def process_media(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.45),
    enable_helmet: bool = Form(True)
):
    if not (is_image(file.filename) or is_video(file.filename)):
        raise HTTPException(status_code=400, detail="Only image/video files are supported")

    output_id = uuid.uuid4().hex[:12]
    ext = file.filename.lower().split(".")[-1]
    upload_path = UPLOAD_DIR / f"{output_id}.{ext}"

    with upload_path.open("wb") as f:
        f.write(await file.read())

    result = process_video(upload_path, output_id, confidence_threshold, enable_helmet) if is_video(file.filename) else process_image(upload_path, output_id, confidence_threshold, enable_helmet)

    original_url = f"/media/uploads/{upload_path.name}"
    annotated_url = f"/media/outputs/{result['annotated_path'].name}"
    plate_crop_url = f"/media/outputs/{result['plate_path'].name}" if result["plate_path"] else None

    db_res = insert_record(
        plate_number=result["plate_number"],
        helmet_status=result["helmet_status"],
        violation_status=result["violation_status"],
        fine_status=result["fine_status"],
        confidence=float(result["confidence"]),
        helmet_conf=float(result["helmet_conf"]),
        plate_conf=float(result["plate_conf"]),
        media_type=result["media_type"],
        original_url=original_url,
        annotated_url=annotated_url,
        plate_crop_url=plate_crop_url,
    )

    return ProcessResult(
        record_id=db_res["record_id"],
        case_id=db_res["case_id"],
        media_type=result["media_type"],
        original_url=original_url,
        annotated_url=annotated_url,
        plate_crop_url=plate_crop_url,
        plate_number=result["plate_number"],
        confidence=float(result["confidence"]),
        helmet_conf=float(result["helmet_conf"]),
        plate_conf=float(result["plate_conf"]),
        helmet_status=result["helmet_status"],
        violation_status=result["violation_status"],
        fine_status=result["fine_status"],
        repeat_offender=db_res["repeat_offender"],
        risk_level=db_res["risk_level"],
        seen_count=db_res["seen_count"],
        created_at=db_res["created_at"],
    )


@app.get("/api/records")
def get_records(
    search: str = Query(default=""),
    violations_only: bool = Query(default=False),
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = "SELECT * FROM records WHERE plate_number LIKE ?"
    params: List = [f"%{search.strip()}%"]
    if violations_only:
        query += " AND violation_status = ?"
        params.append("Violation")
    query += " ORDER BY datetime(created_at) DESC"

    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


@app.get("/api/evidence/{record_id}")
def get_evidence(record_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM records WHERE id = ?", (record_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Record not found")
    return dict(row)


@app.get("/api/analytics")
def get_analytics():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Group by the hour of created_at
    cur.execute("SELECT strftime('%H', created_at) as hour, COUNT(*) as count FROM records WHERE violation_status = 'Violation' GROUP BY hour ORDER BY hour ASC")
    rows = cur.fetchall()
    conn.close()
    
    # Initialize 24-hour array with zeros
    data = [0] * 24
    for row in rows:
        if row[0] is not None:
            hour = int(row[0])
            data[hour] = row[1]
    
    return {"hourly_violations": data}


@app.put("/api/records/{record_id}/fine")
def update_fine_status(record_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT fine_status FROM records WHERE id = ?", (record_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Record not found")
        
    current_status = row[0]
    # Toggle logic
    new_status = "Paid - ₹500" if "Unpaid" in current_status else "Unpaid - ₹500"
    
    cur.execute("UPDATE records SET fine_status = ? WHERE id = ?", (new_status, record_id))
    conn.commit()
    conn.close()
    return {"id": record_id, "fine_status": new_status}


import io
import json
import zipfile
from fastapi.responses import StreamingResponse

@app.get("/api/export/{record_id}")
def export_evidence_pack(record_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM records WHERE id = ?", (record_id,))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Record not found")
        
    record = dict(row)
    
    # Create in-memory zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        # Add metadata json
        metadata = json.dumps(record, indent=4)
        zip_file.writestr(f"{record['case_id']}_metadata.json", metadata)
        
        # Add annotated image/video
        annotated_path = BASE_DIR / record['annotated_url'].lstrip("/media/")
        if annotated_path.exists():
            zip_file.writestr(f"{record['case_id']}_annotated{annotated_path.suffix}", annotated_path.read_bytes())
            
        # Add plate crop
        if record['plate_crop_url']:
            plate_path = BASE_DIR / record['plate_crop_url'].lstrip("/media/")
            if plate_path.exists():
                zip_file.writestr(f"{record['case_id']}_plate{plate_path.suffix}", plate_path.read_bytes())
                
    zip_buffer.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="{record["case_id"]}_evidence.zip"'
    }
    return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)


@app.get("/")
def serve_home():
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(str(BASE_DIR / "frontend" / "dashboard.html"))


@app.get("/rules")
def serve_rules():
    return FileResponse(str(BASE_DIR / "frontend" / "rules.html"))


@app.delete("/api/records/{record_id}")
def delete_record(record_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return {"message": "Record successfully deleted"}


app.mount("/assets", StaticFiles(directory=str(BASE_DIR / "frontend")), name="assets")
app.mount("/media", StaticFiles(directory=str(BASE_DIR)), name="media")


@app.on_event("startup")
def startup() -> None:
    init_db()
