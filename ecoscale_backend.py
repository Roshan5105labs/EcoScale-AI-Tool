"""
EcoScale Backend

"""

import asyncio
import base64
import csv
import json
import os
import random
import time
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import psutil
import cv2
import onnxruntime as ort

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# NVIDIA Power (pynvml) 
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _raw        = pynvml.nvmlDeviceGetName(_gpu_handle)
    GPU_NAME    = _raw.decode() if isinstance(_raw, bytes) else _raw
    print(f"[EcoScale] NVIDIA GPU  : {GPU_NAME}")
except Exception as _e:
    NVML_AVAILABLE = False
    _gpu_handle    = None
    GPU_NAME       = "NVIDIA GPU (pynvml unavailable)"
    print(f"[EcoScale] pynvml unavailable: {_e}")

# WMI for real CPU temperature (Windows) 
try:
    import wmi
    _wmi = wmi.WMI(namespace="root\\OpenHardwareMonitor")
    WMI_AVAILABLE = True
    print("[EcoScale] WMI (OpenHardwareMonitor) : ✓")
except Exception:
    try:
        import wmi
        _wmi = wmi.WMI(namespace="root\\WMI")
        WMI_AVAILABLE = True
        print("[EcoScale] WMI (root\\WMI) : ✓")
    except Exception as _e2:
        WMI_AVAILABLE = False
        _wmi          = None
        print(f"[EcoScale] WMI unavailable: {_e2} — CPU temp will use psutil fallback")

# scikit-learn for policy model 
try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
    SKLEARN_AVAILABLE = True
    print("[EcoScale] scikit-learn : ✓")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[EcoScale] scikit-learn not found — pip install scikit-learn joblib")

#  ONNX Providers 
_avail         = ort.get_available_providers()
CUDA_AVAILABLE = "CUDAExecutionProvider" in _avail
NPU_AVAILABLE  = "VitisAIExecutionProvider" in _avail
print(f"[EcoScale] CUDA provider  : {'✓' if CUDA_AVAILABLE else '✗'}")
print(f"[EcoScale] VitisAI NPU    : {'✓' if NPU_AVAILABLE else '✗'}")

#  Paths
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
YOLO_PATH       = os.path.join(MODELS_DIR, "yolov8n.onnx")
MOBILE_PATH     = os.path.join(MODELS_DIR, "mobilenet_ssd.onnx")
BENCH_FILE      = os.path.join(BASE_DIR, "benchmark_results.json")
TELEMETRY_FILE  = os.path.join(BASE_DIR, "telemetry_log.csv")
POLICY_FILE     = os.path.join(BASE_DIR, "policy_model.joblib")

CO2_GRAMS_PER_WH = 0.82  

COCO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

#  Telemetry feature names (order matters — model trained on this)
FEATURE_NAMES = [
    "battery_pct",      # 0–100
    "drain_rate",       # % per minute, negative = draining
    "gpu_temp",         # °C
    "cpu_temp",         # °C
    "wattage",          # W (NVML or estimated)
    "fps",              # frames per second
    "inference_ms",     # ms per frame
    "plugged",          # 1 = plugged, 0 = battery
    "cpu_usage",        # 0–100 %
    "hour_of_day",      # 0–23
]
# Label: 0 = use CPU/NPU (efficient), 1 = use GPU (performance)



class TelemetryCollector:
    """
    Collects real hardware signals from Windows APIs.
    Battery drain rate is derived from consecutive battery readings.
    CPU temperature is read from WMI / OpenHardwareMonitor.
    GPU temperature is read from pynvml.
    Everything is logged to telemetry_log.csv with timestamps.
    """

    def __init__(self):
        self._lock            = threading.Lock()
        self._batt_history    = deque(maxlen=120)  
        self._last_reading    = {}
        self._simulation_overrides: Dict[str, Any] = {}
        self._cpu_percent_init = False

        # Init CSV log
        if not os.path.exists(TELEMETRY_FILE):
            with open(TELEMETRY_FILE, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "label"] + FEATURE_NAMES)
        print(f"[Telemetry] Logging to {TELEMETRY_FILE}")
        # Prime psutil cpu_percent so first real call returns a value
        psutil.cpu_percent(interval=0.1)

    #Temperature readers 
    def _cpu_temp(self) -> float:
        """
        Read CPU temperature on Windows using 3 fallback methods.
        Method 1: WMI OpenHardwareMonitor (needs OHM running as service)
        Method 2: Windows built-in MSAcpi_ThermalZoneTemperature
        Method 3: Estimate from real CPU usage (always works)
        """
        # Method 1 — OpenHardwareMonitor WMI
        if WMI_AVAILABLE and _wmi:
            try:
                for sensor in _wmi.Sensor():
                    if sensor.SensorType == "Temperature" and "CPU" in sensor.Name:
                        return float(sensor.Value)
            except Exception:
                pass
        # Method 2 — Windows built-in thermal zones (no extra software needed)
        try:
            import wmi as _wmi_local
            w2 = _wmi_local.WMI(namespace="root/WMI")
            for zone in w2.MSAcpi_ThermalZoneTemperature():
                celsius = (zone.CurrentTemperature / 10.0) - 273.15
                if 20.0 < celsius < 120.0:
                    return round(celsius, 1)
        except Exception:
            pass
        # Method 3 — Estimate from CPU usage (always available on Windows)
        # Ryzen 7 7840HS: ~38C idle, ~80C full load
        try:
            usage = psutil.cpu_percent(interval=None)
            return round(38.0 + (usage / 100.0) * 42.0, 1)
        except Exception:
            pass
        return 50.0

    def _gpu_temp(self) -> float:
        if not NVML_AVAILABLE or _gpu_handle is None:
            return -1.0
        try:
            return float(pynvml.nvmlDeviceGetTemperature(
                _gpu_handle, pynvml.NVML_TEMPERATURE_GPU
            ))
        except Exception:
            return -1.0

    def _gpu_wattage(self) -> float:
        if not NVML_AVAILABLE or _gpu_handle is None:
            return -1.0
        try:
            return round(pynvml.nvmlDeviceGetPowerUsage(_gpu_handle) / 1000.0, 2)
        except Exception:
            return -1.0

    # Drain rate calculation
    def _drain_rate(self, current_pct: float, plugged: bool) -> float:
        """
        Calculate battery drain rate in % per minute.
        Negative = draining. Positive = charging. 0 = stable.

        Battery % changes very slowly (~0.05%/min under load).
        We use psutil secsleft for a better real-time estimate,
        falling back to slope calculation over a long window.
        """
        now = time.time()
        self._batt_history.append((now, current_pct, plugged))

        # Primary method: use psutil's time-to-empty/full for real drain rate
        try:
            batt = psutil.sensors_battery()
            if batt and batt.secsleft not in (psutil.POWER_TIME_UNLIMITED,
                                               psutil.POWER_TIME_UNKNOWN, -1, -2):
                if not plugged and batt.secsleft > 0:
                    rate = -(current_pct / (batt.secsleft / 60.0))
                    return round(max(-5.0, rate), 3)
                elif plugged and batt.secsleft > 0:
                    remaining_to_full = 100.0 - current_pct
                    rate = remaining_to_full / (batt.secsleft / 60.0)
                    return round(min(5.0, rate), 3)
        except Exception:
            pass

        # Fallback: slope over history window (works after ~30s)
        if len(self._batt_history) >= 3:
            t_old, pct_old, _ = self._batt_history[0]
            t_new, pct_new, _ = self._batt_history[-1]
            dt_min = (t_new - t_old) / 60.0
            if dt_min >= 0.1:
                return round((pct_new - pct_old) / dt_min, 3)

        # Immediate estimate from plugged state (instant — always shown)
        if not plugged:
            return -0.8   
        elif current_pct < 99:
            return 0.5    
        return 0.0

    #  Main reading 
    def read(self, fps: float = 0.0, inference_ms: float = 0.0,
             active_profile: str = "GPU") -> Dict[str, Any]:
     
        batt   = psutil.sensors_battery()
        plugged    = bool(batt.power_plugged) if batt else True
        batt_pct   = round(batt.percent, 1) if batt else 100.0
        drain_rate = self._drain_rate(batt_pct, plugged)
        gpu_temp   = self._gpu_temp()
        cpu_temp   = self._cpu_temp()
        wattage    = self._gpu_wattage()
        cpu_usage  = psutil.cpu_percent(interval=0.1) 
        hour       = datetime.now().hour

        # If NVML unavailable, estimate wattage from profile
        if wattage < 0:
            base = 45.0 if active_profile == "GPU" else 10.0
            wattage = round(base + random.uniform(-1, 1), 2)

        reading = {
            "battery_pct" : batt_pct,
            "drain_rate"  : drain_rate,
            "gpu_temp"    : gpu_temp if gpu_temp >= 0 else 55.0,
            "cpu_temp"    : cpu_temp if cpu_temp >= 0 else 50.0,
            "wattage"     : wattage,
            "fps"         : round(fps, 1),
            "inference_ms": round(inference_ms, 1),
            "plugged"     : int(plugged),
            "cpu_usage"   : round(cpu_usage, 1),
            "hour_of_day" : hour,
            "_plugged_bool": plugged,
            "_batt_pct"   : batt_pct,
            "_gpu_wattage": wattage,
        }

        # Apply simulation overrides (for demo scenarios)
        with self._lock:
            for k, v in self._simulation_overrides.items():
                if k in reading:
                    reading[k] = v

        with self._lock:
            self._last_reading = dict(reading)

        return reading

    def set_simulation(self, overrides: Dict[str, Any]):
        with self._lock:
            self._simulation_overrides = overrides
        print(f"[Telemetry] Simulation overrides set: {overrides}")

    def clear_simulation(self):
        with self._lock:
            self._simulation_overrides = {}
        print("[Telemetry] Simulation cleared — back to real data")

    def is_simulating(self) -> bool:
        with self._lock:
            return bool(self._simulation_overrides)

    def log_to_csv(self, reading: Dict, label: int):
        try:
            with open(TELEMETRY_FILE, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.utcnow().isoformat(),
                    label,
                ] + [reading.get(feat, 0) for feat in FEATURE_NAMES])
        except Exception as e:
            print(f"[Telemetry] CSV write error: {e}")

    def last(self) -> Dict:
        with self._lock:
            return dict(self._last_reading)

    def csv_row_count(self) -> int:
        if not os.path.exists(TELEMETRY_FILE):
            return 0
        try:
            with open(TELEMETRY_FILE) as f:
                return sum(1 for _ in f) - 1  
        except Exception:
            return 0



class PolicyModel:

    def __init__(self):
        self.model: Optional[Any]   = None
        self.trained_at: Optional[str] = None
        self.n_real_samples: int    = 0
        self.n_synthetic_samples: int = 0
        self.accuracy: float        = 0.0
        self.feature_importances: Dict = {}
        self._lock = threading.Lock()
        self._load_if_exists()

    def _load_if_exists(self):
        if SKLEARN_AVAILABLE and os.path.exists(POLICY_FILE):
            try:
                saved = joblib.load(POLICY_FILE)
                self.model              = saved["model"]
                self.trained_at         = saved.get("trained_at")
                self.n_real_samples     = saved.get("n_real_samples", 0)
                self.n_synthetic_samples= saved.get("n_synthetic_samples", 0)
                self.accuracy           = saved.get("accuracy", 0.0)
                self.feature_importances= saved.get("feature_importances", {})
                print(f"[Policy] Loaded saved model — accuracy {self.accuracy:.1%} "
                      f"({self.n_real_samples} real + {self.n_synthetic_samples} synthetic samples)")
            except Exception as e:
                print(f"[Policy] Could not load saved model: {e}")

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    # Synthetic training data
    def _generate_synthetic(self, n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(42)
        X, y = [], []

        for _ in range(n):
            batt_pct     = rng.uniform(5, 100)
            drain_rate   = rng.uniform(-3.0, 0.5)  
            gpu_temp     = rng.uniform(35, 95)
            cpu_temp     = rng.uniform(35, 90)
            wattage      = rng.uniform(8, 50)
            fps          = rng.uniform(10, 65)
            inference_ms = rng.uniform(10, 150)
            plugged      = rng.randint(0, 2)
            cpu_usage    = rng.uniform(5, 95)
            hour         = rng.randint(0, 24)

            row = [batt_pct, drain_rate, gpu_temp, cpu_temp,
                   wattage, fps, inference_ms, plugged, cpu_usage, hour]


            label = 1   # default: use GPU

            # Thermal protection — GPU too hot
            if gpu_temp > 85:
                label = 0

            # Battery critical — always save power
            if batt_pct < 15:
                label = 0

            # Battery low + draining fast — switch to CPU
            if batt_pct < 40 and drain_rate < -1.5:
                label = 0

            # Battery medium + very fast drain — switch
            if batt_pct < 60 and drain_rate < -2.5:
                label = 0

            # Unplugged + battery below 55% — CPU more efficient
            if plugged == 0 and batt_pct < 55:
                label = 0

            # Plugged in but charging slowly + high temp — reduce load
            if plugged == 1 and drain_rate < -0.5 and gpu_temp > 80:
                label = 0

            # Night time + low activity — power saving
            if (hour >= 23 or hour <= 5) and cpu_usage < 30:
                label = 0

            # FPS already sufficient at low resolution — no need for GPU
            if fps > 28 and inference_ms < 40 and batt_pct < 50:
                label = 0

            # Plugged in, battery stable, normal temp — GPU preferred
            if plugged == 1 and drain_rate > -0.3 and gpu_temp < 75 and batt_pct > 50:
                label = 1

            # High FPS requirement + plugged in — always GPU
            if fps < 15 and plugged == 1 and batt_pct > 30:
                label = 1 

            X.append(row)
            y.append(label)

        return np.array(X), np.array(y)

    # Load real telemetry 
    def _load_real_telemetry(self) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(TELEMETRY_FILE):
            return np.array([]).reshape(0, len(FEATURE_NAMES)), np.array([])

        X, y = [], []
        try:
            with open(TELEMETRY_FILE) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        label = int(row["label"])
                        feats = [float(row[f]) for f in FEATURE_NAMES]
                        X.append(feats)
                        y.append(label)
                    except Exception:
                        continue
        except Exception as e:
            print(f"[Policy] CSV read error: {e}")

        return np.array(X), np.array(y)

    # Train 
    def train(self) -> Dict:
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not installed — pip install scikit-learn joblib"}

        print("[Policy] Starting training…")

        # Real data
        X_real, y_real = self._load_real_telemetry()
        n_real = len(y_real)

        # Synthetic data
        X_syn, y_syn = self._generate_synthetic(2000)

        # Combine — real data gets 3x weight via repetition
        if n_real > 10:
            X_combined = np.vstack([
                np.tile(X_real, (3, 1)),   # real data weighted 3x
                X_syn
            ])
            y_combined = np.concatenate([
                np.tile(y_real, 3),
                y_syn
            ])
            print(f"[Policy] Training on {n_real} real samples (3× weighted) + {len(y_syn)} synthetic")
        else:
            X_combined = X_syn
            y_combined = y_syn
            print(f"[Policy] No real data yet — training on {len(y_syn)} synthetic samples only")
            print("[Policy] Run POST /policy/collect to collect real data from your hardware")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42
        )

        # Decision tree — shallow for explainability and speed
        clf = DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        # Feature importances
        importances = {
            FEATURE_NAMES[i]: round(float(v), 4)
            for i, v in enumerate(clf.feature_importances_)
        }
        importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

        # Print decision tree rules (for judges)
        print("\n[Policy] Decision tree rules:")
        print(export_text(clf, feature_names=FEATURE_NAMES, max_depth=4))
        print(f"\n[Policy] Accuracy on held-out set: {acc:.1%}")
        print(f"[Policy] Feature importances: {importances}")

        # Save
        result = {
            "model"                : clf,
            "trained_at"           : datetime.utcnow().isoformat(),
            "n_real_samples"       : n_real,
            "n_synthetic_samples"  : len(y_syn),
            "accuracy"             : round(acc, 4),
            "feature_importances"  : importances,
        }
        joblib.dump(result, POLICY_FILE)

        with self._lock:
            self.model               = clf
            self.trained_at          = result["trained_at"]
            self.n_real_samples      = n_real
            self.n_synthetic_samples = len(y_syn)
            self.accuracy            = result["accuracy"]
            self.feature_importances = importances

        return {
            "status"               : "trained",
            "accuracy"             : f"{acc:.1%}",
            "n_real_samples"       : n_real,
            "n_synthetic_samples"  : len(y_syn),
            "feature_importances"  : importances,
            "trained_at"           : result["trained_at"],
        }

    # Predict
    def predict(self, reading: Dict) -> Dict:
        """
        Run inference on current telemetry.
        Returns: profile key ("GPU"/"NPU"), confidence, and signal breakdown.
        """
        if not self.is_trained:
            # Smart fallback rule using real signals until model is trained
            plugged  = bool(reading.get("plugged", 1))
            batt     = reading.get("battery_pct", 100)
            drain    = reading.get("drain_rate", 0.0)
            gpu_t    = reading.get("gpu_temp", 55)
            fps      = reading.get("fps", 30)
            hour     = reading.get("hour_of_day", 12)

            key = "GPU"  

            # Thermal protection
            if gpu_t > 85:
                key = "NPU"
            # Battery critical
            elif batt < 15:
                key = "NPU"
            # Battery low and draining fast
            elif batt < 40 and drain < -1.5:
                key = "NPU"
            # Unplugged and below 55%
            elif not plugged and batt < 55:
                key = "NPU"
            # Plugged in, stable, good battery — use GPU
            elif plugged and batt > 50 and gpu_t < 80:
                key = "GPU"
            # Night time low activity
            elif (hour >= 23 or hour <= 5):
                key = "NPU"

            reasons = []
            if gpu_t > 85: reasons.append(f"GPU temp {gpu_t:.0f}°C")
            if batt < 15:  reasons.append(f"Battery critical {batt:.0f}%")
            if drain < -1.5: reasons.append(f"Fast drain {drain:.1f}%/min")
            if not plugged and batt < 55: reasons.append(f"Unplugged {batt:.0f}%")
            if not reasons and key == "GPU": reasons.append("AC power, stable thermals")
            if not reasons: reasons.append("Power saving mode")

            return {
                "decision"      : key,
                "confidence"    : 0.75,
                "confidence_pct": 75.0,
                "mode"          : "smart_fallback",
                "reason"        : " · ".join(reasons),
                "prob_gpu"      : 0.75 if key == "GPU" else 0.25,
                "prob_cpu"      : 0.25 if key == "GPU" else 0.75,
                "signal_scores" : self._signal_scores(reading),
            }

        feats = np.array([[reading.get(f, 0) for f in FEATURE_NAMES]])

        with self._lock:
            proba  = self.model.predict_proba(feats)[0]
            label  = int(self.model.predict(feats)[0])


        key        = "GPU" if label == 1 else "NPU"
        confidence = float(proba[label])

        # Build human-readable reason from dominant signals
        reason = self._explain(reading, key)

        return {
            "decision"      : key,
            "confidence"    : round(confidence, 3),
            "confidence_pct": round(confidence * 100, 1),
            "mode"          : "ai_policy",
            "reason"        : reason,
            "prob_gpu"      : round(float(proba[1]), 3),
            "prob_cpu"      : round(float(proba[0]), 3),
            "signal_scores" : self._signal_scores(reading),
        }

    def _explain(self, r: Dict, decision: str) -> str:
        """Generate a plain-English explanation of the decision."""
        reasons = []
        batt    = r.get("battery_pct", 100)
        drain   = r.get("drain_rate", 0)
        gpu_t   = r.get("gpu_temp", 50)
        plugged = bool(r.get("plugged", 1))

        if gpu_t > 82:
            reasons.append(f"GPU too hot ({gpu_t:.0f}°C)")
        if batt < 20:
            reasons.append(f"Battery critical ({batt:.0f}%)")
        elif batt < 40 and drain < -1.5:
            reasons.append(f"Battery {batt:.0f}% draining at {drain:.1f}%/min")
        elif not plugged and batt < 55:
            reasons.append(f"Unplugged at {batt:.0f}%")
        elif plugged and batt > 50 and gpu_t < 78:
            reasons.append("AC power, stable thermals")

        if decision == "GPU" and not reasons:
            reasons.append("Performance conditions met")
        elif decision == "NPU" and not reasons:
            reasons.append("Power efficiency preferred")

        return " · ".join(reasons) if reasons else f"Policy decided {decision}"

    def _signal_scores(self, r: Dict) -> Dict:
      
        batt  = r.get("battery_pct", 100)
        drain = r.get("drain_rate", 0)
        gpu_t = r.get("gpu_temp", 50)
        cpu_t = r.get("cpu_temp", 50)
        watt  = r.get("wattage", 20)
        fps   = r.get("fps", 30)

        return {
            "battery"    : round(batt, 1),
            "drain_rate" : round(abs(drain) / 3.0 * 100, 1),   # 0–100 pressure
            "gpu_temp"   : round((gpu_t - 35) / 60 * 100, 1),  # 35–95°C → 0–100
            "cpu_temp"   : round((cpu_t - 35) / 55 * 100, 1),
            "wattage"    : round(watt / 50 * 100, 1),
            "fps"        : round(fps / 65 * 100, 1),
        }

    def status(self) -> Dict:
        return {
            "is_trained"          : self.is_trained,
            "trained_at"          : self.trained_at,
            "accuracy"            : self.accuracy,
            "n_real_samples"      : self.n_real_samples,
            "n_synthetic_samples" : self.n_synthetic_samples,
            "feature_importances" : self.feature_importances,
            "model_file"          : os.path.exists(POLICY_FILE),
            "sklearn_available"   : SKLEARN_AVAILABLE,
        }



class PolicyDecider:
    """
    Calls policy_model.predict() on every telemetry reading.
    Switches the inference engine only when the decision changes.
    Enforces a minimum hold time to prevent rapid oscillation.
    """
    MIN_HOLD_SECONDS = 3.0   

    def __init__(self, model: PolicyModel):
        self._model      = model
        self._last_key   = "GPU"
        self._last_switch= 0.0
        self._last_result: Dict = {}
        self._lock       = threading.Lock()
        self._enabled    = True

    def set_enabled(self, v: bool):
        with self._lock:
            self._enabled = v

    def decide(self, reading: Dict, engine: "InferenceEngine") -> Dict:
        """
        Called every WebSocket tick with fresh telemetry.
        Returns the policy result dict including decision + confidence.
        """
        result = self._model.predict(reading)
        new_key = result["decision"]

        with self._lock:
            enabled     = self._enabled
            last_key    = self._last_key
            last_switch = self._last_switch

        if not enabled:
            result["switched"] = False
            result["held"]     = True
            return result

        now = time.time()
        hold_ok = (now - last_switch) >= self.MIN_HOLD_SECONDS

        if new_key != last_key and hold_ok:
            engine.set_profile(new_key)
            with self._lock:
                self._last_key    = new_key
                self._last_switch = now
            result["switched"] = True
            print(f"[Policy] Decision: {last_key} → {new_key} "
                  f"(conf {result['confidence']:.0%}) — {result['reason']}")
        else:
            result["switched"] = False

        with self._lock:
            self._last_result = result

        return result

    def force(self, key: str, engine: "InferenceEngine"):
        engine.set_profile(key)
        with self._lock:
            self._last_key    = key
            self._last_switch = time.time()
        print(f"[Policy] Manual override → {key}")


# Profile definitions 
def build_profiles() -> Dict[str, Dict]:
    if CUDA_AVAILABLE:
        gpu_prov = [("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 4 * 1024 ** 3,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }), "CPUExecutionProvider"]
        gpu_proc, gpu_label = "GPU", f"GPU — {GPU_NAME}"
    else:
        gpu_prov = ["CPUExecutionProvider"]
        gpu_proc, gpu_label = "CPU", "CPU Heavy (no CUDA)"

    if NPU_AVAILABLE:
        npu_prov  = [("VitisAIExecutionProvider", {"config_file": "vaip_config.json"}),
                     "CPUExecutionProvider"]
        npu_proc, npu_label, npu_nm = "NPU", "NPU — AMD XDNA", "AMD XDNA NPU"
    else:
        npu_prov  = ["CPUExecutionProvider"]
        npu_proc, npu_label, npu_nm = "CPU", "CPU — Efficient Threads", "CPU Efficient"

    return {
        "GPU": {
            "name": GPU_NAME, "label": gpu_label, "processor": gpu_proc,
            "providers": gpu_prov, "model_path": YOLO_PATH,
            "model_type": "yolo", "input_size": (640, 640),
            "color": "#FF6B35", "wattage_estimate": 45.0,
            "algorithm": "YOLOv8n — 640×640",
            "why": "CUDA parallelism makes 640×640 matrix ops cheap at ~45W.",
        },
        "NPU": {
            "name": npu_nm, "label": npu_label, "processor": npu_proc,
            "providers": npu_prov, "model_path": MOBILE_PATH,
            "model_type": "mobilenet", "input_size": (300, 300),
            "color": "#00E5A0", "wattage_estimate": 10.0,
            "algorithm": "MobileNet SSD — 300×300",
            "why": "Depthwise convolutions at 300×300 — real-time detection at ~10W.",
        },
    }

PROFILES: Dict[str, Dict] = build_profiles()


# ONNX Session Manager 
class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[ort.InferenceSession]:
        with self._lock:
            return self._sessions.get(key)

    def load_all(self):
        for key, p in PROFILES.items():
            path = p["model_path"]
            if not os.path.exists(path):
                print(f"[Session] ⚠ Model missing: {path}")
                continue
            print(f"[Session] Loading {key} → {os.path.basename(path)}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.enable_mem_pattern = True
            if key == "NPU":
                opts.intra_op_num_threads = min(6, os.cpu_count() or 4)
            sess = ort.InferenceSession(path, sess_options=opts, providers=p["providers"])
            print(f"[Session] {key} active providers: {sess.get_providers()}")
            with self._lock:
                self._sessions[key] = sess


#  Power Monitor (lightweight — telemetry collector does the heavy work)
class PowerMonitor:
    def __init__(self):
        self._plugged: Optional[bool] = None
        self._batt_pct: float = 100.0
        self._callbacks = []
        self._lock = threading.Lock()

    def add_callback(self, fn): self._callbacks.append(fn)

    def poll(self) -> Dict:
        batt = psutil.sensors_battery()
        if batt is None:
            new_plugged, new_pct = True, 100.0
        else:
            new_plugged = batt.power_plugged
            new_pct     = round(batt.percent, 1)

        with self._lock:
            changed = new_plugged != self._plugged
            self._plugged, self._batt_pct = new_plugged, new_pct

        if changed:
            for cb in self._callbacks:
                cb(new_plugged)

        return {"plugged": new_plugged, "battery_pct": new_pct}


# Sustainability Tracker 
class SustainabilityTracker:
    def __init__(self):
        self._start    = time.time()
        self._wh_saved = 0.0
        self._wh_used  = 0.0
        self._last     = time.time()
        self._baseline = PROFILES["GPU"]["wattage_estimate"]
        self._lock     = threading.Lock()

    def set_baseline(self, w: float):
        with self._lock:
            self._baseline = w

    def tick(self, current_w: float, is_gpu: bool):
        now = time.time()
        with self._lock:
            dt = (now - self._last) / 3600.0
            self._last  = now
            self._wh_used += current_w * dt
            if not is_gpu:
                self._wh_saved += max(0, (self._baseline - current_w) * dt)

    def snapshot(self) -> Dict:
        with self._lock:
            wh_s = self._wh_saved
            wh_u = self._wh_used
            elapsed = time.time() - self._start

        co2_g   = wh_s * CO2_GRAMS_PER_WH * 1000
        scale   = (4 * 3600 * 250) / max(elapsed, 1)
        yr_kg   = (co2_g / 1000) * scale
        phone_m = (wh_s / 0.005) * 60 if wh_s > 0 else 0

        return {
            "session_elapsed_s"      : round(elapsed),
            "wh_saved_session"       : round(wh_s * 1000, 2),
            "wh_consumed_session"    : round(wh_u * 1000, 2),
            "co2_saved_session_g"    : round(co2_g, 2),
            "co2_year_projection_kg" : round(yr_kg, 3),
            "phone_charge_minutes"   : round(phone_m, 1),
            "grid_factor"            : CO2_GRAMS_PER_WH,
        }


# Benchmark Logger
class BenchmarkLogger:
    def __init__(self):
        self.running = False
        self.results = self._load()
        self._lock   = threading.Lock()

    def _load(self) -> Dict:
        if os.path.exists(BENCH_FILE):
            try:
                with open(BENCH_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def has_results(self) -> bool:
        return bool(self.results)

    def summary(self) -> Dict:
        with self._lock:
            return dict(self.results)

    def run(self, engine: "InferenceEngine", telemetry: TelemetryCollector,
            duration: int = 30) -> Dict:
        self.running = True
        data = {}

        for key in ["GPU", "NPU"]:
            print(f"[Benchmark] {key} — {duration}s…")
            engine.set_profile(key)
            time.sleep(2)

            w_samples, fps_s, inf_s = [], [], []
            t_end = time.time() + duration

            while time.time() < t_end:
                result  = engine.step()
                reading = telemetry.read(result["fps"], result["inference_ms"], key)
                w = reading["wattage"]
                w_samples.append(w)
                fps_s.append(result["fps"])
                inf_s.append(result["inference_ms"])
                # Log to CSV with ground truth label
                label = 1 if key == "GPU" else 0
                telemetry.log_to_csv(reading, label)
                time.sleep(0.5)

            data[key] = {
                "profile"          : key,
                "algorithm"        : PROFILES[key]["algorithm"],
                "processor"        : PROFILES[key]["processor"],
                "duration_s"       : duration,
                "wattage_mean"     : round(sum(w_samples)/len(w_samples), 2),
                "wattage_min"      : round(min(w_samples), 2),
                "wattage_max"      : round(max(w_samples), 2),
                "fps_mean"         : round(sum(fps_s)/len(fps_s), 1),
                "inference_ms_mean": round(sum(inf_s)/len(inf_s), 1),
                "sample_count"     : len(w_samples),
            }
            print(f"[Benchmark] {key} avg {data[key]['wattage_mean']}W  "
                  f"fps {data[key]['fps_mean']}  "
                  f"inf {data[key]['inference_ms_mean']}ms")

        gpu_w = data["GPU"]["wattage_mean"]
        npu_w = data["NPU"]["wattage_mean"]
        saved = gpu_w - npu_w
        wh_8h = saved * 8.0
        co2_s = wh_8h * CO2_GRAMS_PER_WH
        co2_y = co2_s * 250 / 1000

        results = {
            "recorded_at": datetime.utcnow().isoformat(),
            "gpu"        : data["GPU"],
            "npu"        : data["NPU"],
            "savings": {
                "wattage_saved_w"         : round(saved, 2),
                "pct_reduction"           : round(saved / gpu_w * 100, 1),
                "wh_saved_per_8h_session" : round(wh_8h, 2),
                "co2_saved_per_session_g" : round(co2_s * 1000, 1),
                "co2_saved_per_year_kg"   : round(co2_y, 2),
                "grid_factor"             : "India CEA 2023 — 0.82 kg CO₂/kWh",
                "enterprise_1000_laptops" : {
                    "co2_tonnes_per_year": round(co2_y * 1000 / 1000, 1),
                    "kwh_saved_per_year"  : round(wh_8h * 250 / 1000, 0),
                },
            },
        }

        with self._lock:
            self.results = results
        with open(BENCH_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[Benchmark] Complete — {saved:.1f}W saved  "
              f"{co2_s*1000:.0f}g CO₂/session  → {BENCH_FILE}")
        self.running = False
        return results


#  Presentation Mode 
class PresentationMode:
    def __init__(self):
        self.active           = False
        self._phase_s         = 12
        self._stop            = threading.Event()
        self.current_phase    = "GPU"
        self.phase_remaining  = 0

    def start(self, engine: "InferenceEngine", seconds: int = 12):
        if self.active: return
        self._phase_s = seconds
        self._stop.clear()
        self.active = True
        threading.Thread(target=self._loop, args=(engine,), daemon=True).start()

    def stop(self):
        self._stop.set()
        self.active = False

    def _loop(self, engine):
        idx = 0
        while not self._stop.is_set():
            key = ["GPU","NPU"][idx % 2]
            self.current_phase = key
            engine.set_profile(key)
            for r in range(self._phase_s, 0, -1):
                self.phase_remaining = r
                if self._stop.wait(1): return
            idx += 1

    def status(self) -> Dict:
        return {
            "active"         : self.active,
            "current_phase"  : self.current_phase,
            "phase_remaining": self.phase_remaining,
            "phase_duration" : self._phase_s,
        }


# Inference Engine 
class InferenceEngine:
    def __init__(self, mgr: SessionManager):
        self._mgr       = mgr
        self.active_key = "GPU"
        self._lock      = threading.Lock()
        self._times     = []
        self._cap: Optional[cv2.VideoCapture] = None
        self._open_camera()

    def _open_camera(self):
        for idx in range(5):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 60)
                self._cap = cap
                print(f"[Engine] Webcam @ index {idx}")
                return
        print("[Engine] No webcam — synthetic frames")

    def set_profile(self, key: str):
        with self._lock:
            self.active_key = key

    def _frame(self) -> np.ndarray:
        if self._cap and self._cap.isOpened():
            ret, f = self._cap.read()
            if ret: return f
        return self._synth()

    def _synth(self) -> np.ndarray:
        import math
        t = time.time()
        h, w = 720, 1280
        f = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, 8):
            v = int(70 + 55 * math.sin(t * 1.4 + y * 0.025))
            f[y:y+8, :] = [v//4, v//2, v]
        cx = int(w/2 + 200*math.sin(t*0.6))
        cy = int(h/2 + 80*math.cos(t*0.45))
        cv2.rectangle(f, (cx-80,cy-120),(cx+80,cy+120),(0,200,80),2)
        cv2.putText(f,"person 0.93",(cx-76,cy-130),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,80),2)
        return f

    def _pre_yolo(self, frame, size=(640,640)):
        img = cv2.resize(frame, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        return np.transpose(img,(2,0,1))[np.newaxis]

    def _post_yolo(self, out, oh, ow, thresh=0.45):
        dets, data = [], out[0].T
        sx, sy = ow/640, oh/640
        cls_ids = np.argmax(data[:,4:],axis=1)
        confs   = data[np.arange(len(data)),4+cls_ids]
        for box,conf,cls in zip(data[:,:4],confs,cls_ids):
            if conf<thresh: continue
            cx,cy,bw,bh=box
            dets.append((int((cx-bw/2)*sx),int((cy-bh/2)*sy),
                         int((cx+bw/2)*sx),int((cy+bh/2)*sy),float(conf),int(cls)))
        return dets

    def _pre_mobile(self, frame):
        img = cv2.resize(frame,(300,300))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
        return ((img-127.5)/127.5)[np.newaxis]

    def _post_mobile(self, outputs, oh, ow, thresh=0.4):
        dets=[]
        try:
            boxes=outputs[0][0];cls_ids=outputs[1][0]
            confs=outputs[2][0];num=int(outputs[3][0])
            for i in range(min(num,len(confs))):
                if confs[i]<thresh: continue
                y1,x1,y2,x2=boxes[i]
                dets.append((int(x1*ow),int(y1*oh),int(x2*ow),int(y2*oh),
                              float(confs[i]),int(cls_ids[i])))
        except Exception: pass
        return dets

    def _draw(self, frame, dets, bgr):
        for x1,y1,x2,y2,conf,cls in dets:
            cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
            lbl=f"{COCO_CLASSES[cls] if cls<len(COCO_CLASSES) else cls} {conf:.2f}"
            cv2.putText(frame,lbl,(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,bgr,2)
        return frame

    def step(self) -> Dict[str, Any]:
        with self._lock:
            key = self.active_key

        profile = PROFILES[key]
        sess    = self._mgr.get(key)
        frame   = self._frame()
        h, w    = frame.shape[:2]
        dets, err = [], None

        t0 = time.perf_counter()
        if sess is not None:
            try:
                iname = sess.get_inputs()[0].name
                if profile["model_type"] == "yolo":
                    outs = sess.run(None,{iname:self._pre_yolo(frame,profile["input_size"])})
                    dets = self._post_yolo(outs[0],h,w)
                else:
                    outs = sess.run(None,{iname:self._pre_mobile(frame)})
                    dets = self._post_mobile(outs,h,w)
            except Exception as e:
                err=str(e)
        else:
            time.sleep(1/30)
        t1 = time.perf_counter()

        self._times.append(t1-t0)
        if len(self._times)>30: self._times.pop(0)
        avg = sum(self._times)/len(self._times)
        fps = round(1.0/avg,1) if avg>0 else 0.0

        bgr=(53,107,255) if key=="GPU" else (160,229,0)
        annotated=self._draw(frame.copy(),dets,bgr)
        _,jpeg=cv2.imencode(".jpg",annotated,[cv2.IMWRITE_JPEG_QUALITY,70])
        frame_b64=base64.b64encode(jpeg.tobytes()).decode()

        return {
            "fps"           : fps,
            "inference_ms"  : round((t1-t0)*1000,2),
            "detections"    : len(dets),
            "resolution"    : f"{w}x{h}",
            "algorithm"     : profile["algorithm"],
            "algorithm_why" : profile["why"],
            "processor"     : profile["processor"],
            "profile_name"  : profile["name"],
            "profile_label" : profile["label"],
            "color"         : profile["color"],
            "session_active": sess is not None,
            "frame_b64"     : frame_b64,
            "error"         : err,
        }

    def cleanup(self):
        if self._cap: self._cap.release()


# Application Bootstrap 
session_mgr = SessionManager()
power_mon   = PowerMonitor()
telemetry   = TelemetryCollector()
policy_model= PolicyModel()
decider     = PolicyDecider(policy_model)
sustain     = SustainabilityTracker()
bench       = BenchmarkLogger()
pres        = PresentationMode()
engine: Optional[InferenceEngine] = None

app = FastAPI(title="EcoScale v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    global engine
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, session_mgr.load_all)
    engine = InferenceEngine(session_mgr)
    if bench.has_results():
        r = bench.summary()
        sustain.set_baseline(r["gpu"]["wattage_mean"])
    print("[EcoScale] ✓ v3 Ready")


#  REST Endpoints 
@app.get("/health")
def health():
    return {
        "cuda":CUDA_AVAILABLE,"npu":NPU_AVAILABLE,"nvml":NVML_AVAILABLE,
        "wmi":WMI_AVAILABLE,"gpu_name":GPU_NAME,
        "policy":policy_model.status(),
        "telemetry_rows":telemetry.csv_row_count(),
        "benchmark_available":bench.has_results(),
    }



@app.post("/policy/collect")
async def collect_telemetry(background_tasks: BackgroundTasks, seconds: int = 120):
    """
    Runs both profiles for `seconds` each, logging every reading to
    telemetry_log.csv with correct ground truth labels.
    Calls POST /policy/train.
    """
    if bench.running:
        return {"status":"benchmark_running"}

    def _collect():
        for key in ["GPU","NPU"]:
            print(f"[Collect] {key} — {seconds}s of real telemetry…")
            engine.set_profile(key)
            time.sleep(2)
            label   = 1 if key == "GPU" else 0
            t_end   = time.time() + seconds

            while time.time() < t_end:
                result  = engine.step()
                reading = telemetry.read(result["fps"], result["inference_ms"], key)
                telemetry.log_to_csv(reading, label)
                time.sleep(0.5)

            print(f"[Collect] {key} done — {telemetry.csv_row_count()} total rows")

    background_tasks.add_task(_collect)
    return {"status":"started", "seconds_per_profile":seconds,
            "file":TELEMETRY_FILE}


@app.post("/policy/train")
async def train_policy():
    """Train the policy model on collected real telemetry + synthetic data."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, policy_model.train)
    return result


@app.get("/policy/status")
def policy_status():
    return policy_model.status()


@app.get("/policy/telemetry")
def telemetry_preview():
    """Return last 20 rows of real telemetry for dashboard proof panel."""
    rows = []
    if os.path.exists(TELEMETRY_FILE):
        try:
            with open(TELEMETRY_FILE) as f:
                reader = list(csv.DictReader(f))
                rows   = reader[-20:]   # last 20 readings
        except Exception:
            pass
    return {"rows": rows, "total": telemetry.csv_row_count()}



@app.post("/simulate/conditions")
def simulate_conditions(scenario: str = "real"):
    """
    Inject fake telemetry overrides to demonstrate policy intelligence.
    The AI makes decisions based on these signals — not the cable state.

    Scenarios:
      real          — clear all overrides, use real hardware data
      fast_drain    — battery draining fast → forces CPU even if plugged in
      high_temp     — GPU overheating → forces CPU for thermal protection
      critical_batt — battery critical → forces minimum power
      night_mode    — night + low activity → efficient mode
      peak_demand   — plugged in, stable, high FPS needed → GPU preferred
    """
    scenarios = {
        "real": {},
        "fast_drain": {
            "battery_pct" : 52.0,
            "drain_rate"  : -2.8,
            "plugged"     : 1,         
            "gpu_temp"    : 68.0,
        },
        "high_temp": {
            "gpu_temp"    : 88.0,
            "cpu_temp"    : 82.0,
            "battery_pct" : 70.0,
            "plugged"     : 1,
        },
        "critical_batt": {
            "battery_pct" : 12.0,
            "drain_rate"  : -1.2,
            "plugged"     : 0,
        },
        "night_mode": {
            "hour_of_day" : 2,
            "cpu_usage"   : 15.0,
            "battery_pct" : 45.0,
            "plugged"     : 0,
        },
        "peak_demand": {
            "battery_pct" : 85.0,
            "drain_rate"  : 0.1,        
            "plugged"     : 1,
            "gpu_temp"    : 62.0,
            "fps"         : 12.0,       
        },
    }

    if scenario not in scenarios:
        return {"error": f"Unknown scenario. Choose from: {list(scenarios.keys())}"}

    overrides = scenarios[scenario]
    if scenario == "real":
        telemetry.clear_simulation()
        decider.set_enabled(True)
    else:
        telemetry.set_simulation(overrides)
        decider.set_enabled(True)

    return {
        "scenario" : scenario,
        "overrides": overrides,
        "simulating": telemetry.is_simulating(),
    }


@app.post("/simulate/{mode}")
def force_mode(mode: str):
    """Manual hardware override — bypasses policy model."""
    if mode not in ("gpu","npu"):
        return {"error":"use gpu or npu"}
    decider.set_enabled(False)
    decider.force(mode.upper(), engine)
    return {"active":mode.upper(), "policy_paused":True}


@app.post("/policy/resume")
def resume_policy():
    """Re-enable policy model after manual override."""
    decider.set_enabled(True)
    return {"status":"policy_resumed"}


@app.post("/benchmark")
async def run_benchmark(background_tasks: BackgroundTasks, duration: int = 30):
    if bench.running:
        return {"status":"already_running"}

    def _run():
        results = bench.run(engine, telemetry, duration)
        sustain.set_baseline(results["gpu"]["wattage_mean"])

    background_tasks.add_task(_run)
    return {"status":"started","duration_per_profile_s":duration}


@app.get("/benchmark/results")
def benchmark_results():
    if not bench.has_results():
        return {"status":"no_results","hint":"POST /benchmark first"}
    return bench.summary()


@app.post("/presentation/start")
def pres_start(phase_seconds: int = 12):
    pres.start(engine, phase_seconds)
    return {"status":"started","phase_seconds":phase_seconds}


@app.post("/presentation/stop")
def pres_stop():
    pres.stop()
    return {"status":"stopped"}


#  WebSocket Stream 
@app.websocket("/ws/metrics")
async def stream(ws: WebSocket):
    await ws.accept()
    print("[EcoScale] Dashboard connected")
    loop = asyncio.get_event_loop()

    try:
        while True:
            power_data = power_mon.poll()
            workload   = await loop.run_in_executor(None, engine.step)

            # Collect real telemetry
            reading    = telemetry.read(
                workload["fps"], workload["inference_ms"], engine.active_key
            )

            # Inject real wattage into workload
            workload["wattage"] = reading["wattage"]

            # Run AI policy decision 
            if not pres.active:
                policy_result = decider.decide(reading, engine)
            else:
                policy_result = {"decision": pres.current_phase,
                                 "mode": "presentation",
                                 "confidence": 1.0, "confidence_pct": 100,
                                 "reason": "Presentation mode active",
                                 "switched": False, "signal_scores": {}}

            workload["cpu_pct"] = reading["cpu_usage"]

            # Log this to CSV (label = current active profile)
            label = 1 if engine.active_key == "GPU" else 0
            telemetry.log_to_csv(reading, label)

            #  sustainability
            sustain.tick(reading["wattage"], engine.active_key == "GPU")

            payload = {
                "timestamp"     : datetime.utcnow().isoformat(),
                "power"         : power_data,
                "workload"      : workload,
                "telemetry"     : {k:v for k,v in reading.items()
                                   if not k.startswith("_")},
                "policy"        : policy_result,
                "sustainability": sustain.snapshot(),
                "presentation"  : pres.status(),
                "benchmark"     : bench.summary() if bench.has_results() else None,
                "simulating"    : telemetry.is_simulating(),
                "hardware"      : {
                    "gpu_name": GPU_NAME,
                    "cuda": CUDA_AVAILABLE, "npu": NPU_AVAILABLE,
                    "nvml": NVML_AVAILABLE, "wmi": WMI_AVAILABLE,
                },
            }
            await ws.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        print("[EcoScale] Dashboard disconnected")
    except Exception as e:
        print(f"[EcoScale] WS error: {e}")
    finally:
        if engine: engine.cleanup()


if __name__ == "__main__":
    print("=" * 65)
    print("  EcoScale v3 — AI Policy Edition")
    print("=" * 65)
    print(f"  GPU (CUDA)     : {'✓ '+GPU_NAME if CUDA_AVAILABLE else '✗'}")
    print(f"  NPU (VitisAI)  : {'✓ AMD XDNA' if NPU_AVAILABLE else '✗'}")
    print(f"  NVML Wattage   : {'✓ Real' if NVML_AVAILABLE else '✗ Estimated'}")
    print(f"  WMI CPU Temp   : {'✓ Real' if WMI_AVAILABLE else '✗ Unavailable'}")
    print(f"  scikit-learn   : {'✓' if SKLEARN_AVAILABLE else '✗ pip install scikit-learn joblib'}")
    print(f"  Policy Model   : {'✓ Trained' if policy_model.is_trained else '— POST /policy/collect then /policy/train'}")
    print(f"  Real Telemetry : {telemetry.csv_row_count()} rows in telemetry_log.csv")
    print(f"  Benchmark      : {'✓ Found' if bench.has_results() else '— POST /benchmark'}")
    print("=" * 65)
    print("  Workflow to get real data:")
    print("    1. POST /policy/collect?seconds=120  (2 min per profile)")
    print("    2. POST /policy/train                (train on your data)")
    print("    3. GET  /policy/status               (verify accuracy)")
    print("    4. POST /simulate/conditions?scenario=fast_drain  (demo)")
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

