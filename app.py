
"""
app.py  —  Demo Flask service for iceberg detection & drift forecast.
Run: python app.py
"""
import io, base64, math
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from lib import StatoilCNN2, Berg, integrate, great_circle_km

app = Flask(__name__)

INC_MEDIAN = 39.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clf = StatoilCNN2().to(DEVICE)
clf.load_state_dict(torch.load("results/statoil_cnn_best.pt", map_location=DEVICE))
clf.eval()

yolo = YOLO("runs/iceberg/weights/best.pt")


def is_sar_like(img_bytes: bytes) -> bool:
    """
    Эвристика: SAR-снимки из датасета Statoil имеют низкое среднее
    (тёмный фон океана) и умеренный STD.
    Обычные фото — яркие и контрастные.
    Пороги подобраны по статистике датасета:
      mean_r < 80  (SAR-снимки серые и тёмные)
      std_r  < 70  (нет ярких насыщенных областей)
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((75, 75))
    arr = np.array(img, dtype=np.float32)
    mean_r = arr[:, :, 0].mean()
    std_r  = arr[:, :, 0].std()
    return (mean_r < 80) and (std_r < 70)


def png_to_tensor(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((75, 75))
    arr = np.array(img, dtype=np.float32)
    band1 = arr[:, :, 0]
    band2 = arr[:, :, 1]
    ratio = band1 - band2
    X = np.stack([band1, band2, ratio], axis=0)[None]
    X = (X - X.mean(axis=(2, 3), keepdims=True)) / (X.std(axis=(2, 3), keepdims=True) + 1e-6)
    return torch.tensor(X, dtype=torch.float32)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "no image"}), 400

    raw = f.read()
    img_b64 = base64.b64encode(raw).decode()

    # ── Проверка: не SAR-снимок → объектов нет ──
    if not is_sar_like(raw):
        return jsonify({
            "img_b64":     img_b64,
            "label":       "Объектов не обнаружено",
            "prob":        None,
            "is_iceberg":  False,
            "detections":  [],
            "drift_track": [],
            "drift_km":    None,
        })

    # ── 2. Классификация ──
    X = png_to_tensor(raw).to(DEVICE)
    inc = torch.tensor([INC_MEDIAN], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(clf(X, inc)).item()
    is_iceberg = prob >= 0.5
    label = "Айсберг" if is_iceberg else "Корабль"

    # ── 3. Детекция YOLO ──
    img_pil = Image.open(io.BytesIO(raw)).convert("RGB")
    results = yolo(img_pil, verbose=False)[0]
    detections = []
    for box in results.boxes:
        detections.append({
            "cls":      int(box.cls[0]),
            "cls_name": yolo.names[int(box.cls[0])],
            "conf":     round(float(box.conf[0]), 3),
            "xyxy":     [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })

    # ── 4. Прогноз дрейфа ──
    drift_track = []
    drift_km = None
    if is_iceberg:
        lat0   = float(request.form.get("Начальная широта айсберга", 75.0))
        lon0   = float(request.form.get("Начальная долгота айсберга", 30.0))
        wind_u = float(request.form.get("Скорость ветра по оси запад–восток", 5.0))
        wind_v = float(request.form.get("Скорость ветра по оси юг–север", 2.0))
        curr_u = float(request.form.get("Скорость течения по оси запад–восток", 0.3))
        curr_v = float(request.form.get("Скорость течения по оси юг–север", 0.1))
        hours  = int(request.form.get("Время прогнозирования дрейфа", 120))

        berg    = Berg(lat=lat0, lon=lon0)
        wind_fn = lambda la, lo, t: (wind_u, wind_v)
        curr_fn = lambda la, lo, t: (curr_u, curr_v)
        track   = integrate(berg, wind_fn, curr_fn, hours=hours)
        drift_track = [
            {"h": int(row[0]), "lat": round(float(row[1]), 4), "lon": round(float(row[2]), 4)}
            for row in track
        ]
        end      = track[-1]
        drift_km = round(great_circle_km(lat0, lon0, float(end[1]), float(end[2])), 1)

    return jsonify({
        "img_b64":     img_b64,
        "label":       label,
        "prob":        round(prob, 3),
        "is_iceberg":  is_iceberg,
        "detections":  detections,
        "drift_track": drift_track,
        "drift_km":    drift_km,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)