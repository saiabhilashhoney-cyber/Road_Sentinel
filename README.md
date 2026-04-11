# 🚦 Road Sentinel – AI-Powered Traffic Violation Detection System

## 🌍 Overview

**Road Sentinel** is a full-stack, AI-powered traffic monitoring system designed for police and administrative workflows.

It detects:

* Helmet violations
* Number plates
* Vehicle compliance

and automatically generates **evidence-backed violation records** using computer vision and OCR.

This project demonstrates **real-world AI deployment**, **computer vision pipelines**, and **end-to-end system integration**.

---

## 🧠 Core Idea

Instead of manual monitoring, Road Sentinel automates traffic enforcement using:

* Object detection (YOLO models)
* Optical Character Recognition (OCR)
* Evidence generation system
* Admin dashboard for tracking violations

---

## 🏗️ System Architecture

```
User Upload (Image / Video)
        ↓
YOLO Detection Pipeline
        ↓
Helmet Detection + Plate Detection
        ↓
OpenCV Processing
        ↓
OCR Extraction (EasyOCR)
        ↓
Violation Decision Engine
        ↓
Database Storage (SQLite)
        ↓
Admin Dashboard (Search + Evidence View)
```

---

## ⚙️ Core Modules

### 📥 Input Handler

* Accepts image or video uploads
* Supports real-time processing via web UI

---

### 🧠 Detection Engine

* Uses YOLO models:

  * `helmet_model.pt` → helmet / no-helmet detection
  * `plate_model.pt` → number plate detection

* Outputs:

  * Bounding boxes
  * Class labels

---

### 🎯 Visual Processing (OpenCV)

* Draws bounding boxes and annotations
* Generates:

  * Annotated images/videos
  * Plate crop images

---

### 🔍 OCR Engine

* Uses **EasyOCR**
* Extracts number plate text
* Handles multiple formats and variations

---

### ⚖️ Violation Decision Engine

* Determines:

  * Helmet violation (Yes/No)
  * Fine status
  * Repeat offender flag

* Generates structured violation records

---

### 🗄️ Data Storage

* SQLite database (`road_sentinel.db`)
* Stores:

  * Plate number
  * Violation status
  * Evidence paths
  * Timestamp

---

### 📊 Admin Dashboard

* Search by plate number
* Filter violation records
* View:

  * Original image
  * Annotated output
  * Plate crop

---

## 🔄 Processing Flow

1. User uploads image/video
2. YOLO detects helmet and number plate
3. OpenCV annotates visual output
4. OCR extracts plate number
5. System determines violation status
6. Data stored in SQLite
7. Dashboard displays results and evidence

---

## 🎨 Frontend Features

* Upload interface for image/video
* Real-time result display
* Violation status card
* Interactive admin dashboard
* Evidence preview system

---

## ⚙️ Tech Stack

### 🔹 Backend

* FastAPI
* Python

### 🔹 Computer Vision

* YOLO (Ultralytics)
* OpenCV

### 🔹 OCR

* EasyOCR

### 🔹 Frontend

* HTML
* CSS
* JavaScript

### 🔹 Database

* SQLite

---

## 📁 Project Structure

```
backend/app.py              # FastAPI server and CV pipeline
frontend/index.html        # Upload UI
frontend/dashboard.html    # Admin dashboard
frontend/css/styles.css    # Styling
frontend/js/app.js         # Upload + results logic
frontend/js/dashboard.js   # Dashboard interactions
models/                    # YOLO model weights
uploads/                   # Uploaded files
outputs/                   # Annotated outputs + crops
road_sentinel.db           # SQLite database
```

---

## 🔐 Requirements

Ensure the following files exist:

* `models/helmet_model.pt`
* `models/plate_model.pt`

---

## ▶️ How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Start backend server

```bash
uvicorn backend.app:app --reload
```

---

### Access application

* Home: http://127.0.0.1:8000/
* Dashboard: http://127.0.0.1:8000/dashboard

---

## 🧪 Sample Workflow

1. Upload traffic image/video
2. System detects:

   * Helmet status
   * Number plate
3. OCR extracts plate number
4. Violation decision is generated
5. Evidence stored and displayed in dashboard

---

## 🚫 Limitations

* Accuracy depends on model quality
* OCR may fail for unclear plates
* No real-time camera integration
* Uses local SQLite (not scalable for production)

---

## 🔥 Future Improvements

* Real-time CCTV integration
* Cloud database (PostgreSQL)
* Automated fine generation system
* Multi-lane traffic monitoring
* Deep learning model improvements

---

## 🏆 Key Highlights

* End-to-end AI-powered violation detection
* YOLO-based object detection pipeline
* OCR-based number plate recognition
* Evidence-based dashboard system
* Full-stack implementation

---

## ⭐ Conclusion

Road Sentinel demonstrates how **Computer Vision + OCR + Backend Systems** can be integrated to build a scalable and intelligent traffic monitoring solution for real-world enforcement systems.
