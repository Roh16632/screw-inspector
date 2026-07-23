from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import StreamingResponse, FileResponse
import cv2
from ultralytics import YOLO
import shutil
import asyncio
import random
import numpy as np

app = FastAPI(title="Digital Twin Vision Inspector")


MODEL_PATH = r'C:\ScrewProject\weights\best.pt'
try:
    model = YOLO(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")



dashboard_data = {
    "machine_status": "Operational",
    "defects_detected": 0,
    "inspection_accuracy": 100,
    "digital_twin_sync": "Active"
}

alerts = [
    {"message": "System Started"}
]


@app.get("/")
def home():
    return FileResponse("static/dtvi.html")


@app.get("/dashboard")
def dashboard():
    return dashboard_data


@app.get("/alerts")
def get_alerts():
    return alerts


@app.post("/detect-frame")
async def detect_frame(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.4)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class": cls_name, "confidence": round(conf, 2)
        })

    defect_count = sum(1 for d in detections if d["class"] != "OK")
    ok_count = len(detections) - defect_count

    dashboard_data["defects_detected"] = defect_count

    for d in detections:
        if d["class"] != "OK":
            alerts.append({"message": f"Defect detected: {d['class']} ({d['confidence']} confidence)"})
    if len(alerts) > 20:
        del alerts[0:len(alerts) - 20]

    return {"detections": detections, "ok_count": ok_count, "defect_count": defect_count}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    try:
        while True:

            sensor_data = {
                "machine_status": dashboard_data["machine_status"],
                "defects_detected": dashboard_data["defects_detected"],
                "alerts": alerts[-5:]
            }

            await websocket.send_json(sensor_data)

            await asyncio.sleep(1)

    except Exception:
        pass
