# Road Sentinel

Road Sentinel is a full-stack traffic violation dashboard for police/admin workflows.
It detects helmet violations, detects number plates, extracts OCR text, and stores evidence.

## Features

- Upload image or video from web UI
- YOLO inference:
  - `helmet_model.pt` for helmet/no-helmet detection
  - `plate_model.pt` for number plate detection
- OpenCV visual overlays:
  - Bounding boxes and labels (`Helmet`, `No Helmet`, `Number Plate`)
  - Annotated output image/video generation
- EasyOCR plate number extraction
- Violation decision card (safe/violation, fine status, repeat offender flag)
- Admin dashboard:
  - Search by plate
  - Filter violations only
  - View evidence (original, annotated, plate crop)
- SQLite persistence (`road_sentinel.db`)

## Project Structure

- `backend/app.py` - FastAPI server, CV pipeline, SQLite APIs
- `frontend/index.html` - Upload and live results page
- `frontend/dashboard.html` - Records admin table and evidence viewer
- `frontend/css/styles.css` - Theme and dashboard styling
- `frontend/js/app.js` - Upload flow and result rendering
- `frontend/js/dashboard.js` - Records + evidence interactions
- `models/` - trained model weights
- `uploads/` - uploaded files
- `outputs/` - annotated outputs and plate crops

## Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start server:

```bash
uvicorn backend.app:app --reload
```

3. Open:

- `http://127.0.0.1:8000/` for Home
- `http://127.0.0.1:8000/dashboard` for Dashboard

## Notes

- Ensure model files exist:
  - `models/helmet_model.pt`
  - `models/plate_model.pt`
- EasyOCR first run may download language assets automatically.
