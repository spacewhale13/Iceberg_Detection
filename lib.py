"""Utilities: SAR preprocessing, metrics, drift integrator."""
from __future__ import annotations
import json, time, math
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support

# ---------- IO ----------
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

def save_json(obj, name):
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"[saved] {p}")

# ---------- Statoil preprocessing ----------
def statoil_to_tensor(df):
    """df: pandas DataFrame from Statoil train.json. Returns X (N,3,75,75), y (N,), inc (N,)."""
    band1 = np.stack(df["band_1"].apply(np.array).values).reshape(-1, 75, 75)
    band2 = np.stack(df["band_2"].apply(np.array).values).reshape(-1, 75, 75)
    # 3rd channel: ratio in dB
    ratio = band1 - band2
    X = np.stack([band1, band2, ratio], axis=1).astype(np.float32)
    # per-image normalization
    X = (X - X.mean(axis=(2,3), keepdims=True)) / (X.std(axis=(2,3), keepdims=True) + 1e-6)
    y = df["is_iceberg"].values.astype(np.int64) if "is_iceberg" in df.columns else None
    inc = df["inc_angle"].replace("na", np.nan).astype(float).fillna(df["inc_angle"].replace("na", np.nan).astype(float).median()).values.astype(np.float32)
    return X, y, inc

# ---------- Statoil model ----------
class StatoilCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(block(3,32), block(32,64), block(64,128), block(128,256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256+1,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
    def forward(self, x, inc):
        z = self.f(x)
        z = nn.functional.adaptive_avg_pool2d(z,1).flatten(1)
        z = torch.cat([z, inc.unsqueeze(1)], dim=1)
        return self.head[2:](self.head[1](self.head[0](z.unsqueeze(-1).unsqueeze(-1))))[:,0] if False else self._head(z)
    def _head(self, z):
        x = nn.functional.relu(self.head[2](z))
        x = self.head[3](x)
        return self.head[4](x).squeeze(1)

# Cleaner version
class StatoilCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                              nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.feat = nn.Sequential(block(3,32), block(32,64), block(64,128), block(128,256), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(257,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,1))
    def forward(self,x,inc):
        z = self.feat(x)
        return self.head(torch.cat([z, inc.unsqueeze(1)], dim=1)).squeeze(1)

# ---------- Metrics ----------
def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(p), "recall": float(r),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "n": int(len(y_true)), "threshold": threshold,
    }

# ---------- Drift: Bigg 1997 simplified ----------
@dataclass
class Berg:
    lat: float; lon: float
    L: float = 200.0   # length, m
    W: float = 200.0   # width, m
    H: float = 100.0   # height total, m
    rho_i: float = 900.0
    rho_w: float = 1025.0
    rho_a: float = 1.225
    Ca: float = 1.3; Cw: float = 0.9
    @property
    def submerged_frac(self): return self.rho_i/self.rho_w
    @property
    def draft(self): return self.H * self.submerged_frac
    @property
    def freeboard(self): return self.H - self.draft
    @property
    def Aa(self): return self.L * self.freeboard
    @property
    def Aw(self): return self.L * self.draft
    @property
    def mass(self): return self.rho_i * self.L * self.W * self.H

def coriolis_f(lat_deg): return 2*7.2921e-5*math.sin(math.radians(lat_deg))

def step_rk4(state, berg, wind_uv_fn, curr_uv_fn, t, dt):
    def deriv(s, t):
        lat, lon, vx, vy = s
        lat = max(-89.9, min(89.9, lat))           # guard
        Ua = np.array(wind_uv_fn(lat, lon, t))
        Uw = np.array(curr_uv_fn(lat, lon, t))
        V = np.array([vx, vy])
        relA = Ua - V; relW = Uw - V
        Fa = 0.5*berg.rho_a*berg.Ca*berg.Aa * np.linalg.norm(relA)*relA
        Fw = 0.5*berg.rho_w*berg.Cw*berg.Aw * np.linalg.norm(relW)*relW
        f = coriolis_f(lat)
        Fc = -berg.mass*f*np.array([-vy, vx])
        a = (Fa + Fw + Fc)/berg.mass
        dlat = vy/111_000.0
        dlon = vx/(111_000.0*math.cos(math.radians(lat)) + 1e-9)
        return np.array([dlat, dlon, a[0], a[1]])
    k1 = deriv(state, t)
    k2 = deriv(state + dt/2*k1, t + dt/2)
    k3 = deriv(state + dt/2*k2, t + dt/2)
    k4 = deriv(state + dt*k3, t + dt)
    new = state + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    # hard guards: lat, sane velocities
    new[0] = max(-89.9, min(89.9, new[0]))
    new[2] = max(-5.0, min(5.0, new[2]))           # m/s
    new[3] = max(-5.0, min(5.0, new[3]))
    if not np.all(np.isfinite(new)):
        new = state                                # zero-step fallback
    return new

def integrate(berg, wind_uv_fn, curr_uv_fn, hours=120, dt=600):   # было 3600
    s = np.array([berg.lat, berg.lon, 0.0, 0.0])
    track = [(0.0, s[0], s[1])]
    t = 0.0
    steps_per_hour = int(3600/dt)
    for h in range(hours):
        for _ in range(steps_per_hour):
            s = step_rk4(s, berg, wind_uv_fn, curr_uv_fn, t, dt)
            t += dt
        track.append((h+1, s[0], s[1]))
    return np.array(track)

def great_circle_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(min(1, math.sqrt(a)))