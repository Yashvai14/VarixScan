# backend/main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
import uvicorn
import random
from datetime import datetime

app = FastAPI()

# CORS (to allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy ML model (replace later with trained CNN)
def analyze_image(file_path: str):
    # Replace with real model prediction
    classes = ["No Varicose Veins", "Mild", "Moderate", "Severe"]
    diagnosis = random.choice(classes)
    confidence = round(random.uniform(70, 95), 2)
    return diagnosis, confidence

@app.post("/analyze")
async def analyze(
    file: UploadFile,
    name: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
):
    save_path = f"uploads/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    diagnosis, confidence = analyze_image(save_path)

    # TODO: Save to DB
    patient_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "image": save_path,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    return patient_data

# Temporary in-memory storage
reports = []
report_id = 1

@app.post("/analyze")
async def analyze(
    file: UploadFile,
    name: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
):
    global report_id
    save_path = f"uploads/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    diagnosis, confidence = analyze_image(save_path)

    report = {
        "id": report_id,
        "name": name,
        "age": age,
        "gender": gender,
        "image": save_path,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    reports.append(report)
    report_id += 1

    return report


@app.get("/reports")
def get_reports():
    return reports

