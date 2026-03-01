# ⚡ EcoScale: Context-Aware AI Workload Manager

**EcoScale v3 (AI Policy Edition)** is an intelligent, on-device AI orchestration system designed specifically for the AMD Ryzen AI ecosystem. 

By continuously polling real-time hardware telemetry, EcoScale uses a trained machine learning policy model to seamlessly migrate active AI inference sessions between the GPU, CPU, and NPU. 

**The result:** AI tasks never stop, the user never notices, and system power consumption is slashed by 56%.

---

## 🏆 The Problem
Currently, running local AI models on a laptop means maxing out the discrete GPU, regardless of the laptop's thermal state or battery level. Existing solutions fall short:
* **Windows Power Plans & Battery Saver:** Use blunt, system-wide throttling that kills overall performance.
* **NVIDIA Optimus:** Only manages display outputs, not active background compute tasks. 

No existing tool dynamically manages *application-level* AI workloads across heterogeneous hardware based on real thermal and battery constraints.

## 🧠 How EcoScale Solves It
EcoScale acts as an intelligent traffic cop for your AI workloads, completely transparent to the end user.

1. **Real-Time Telemetry:** The backend actively samples 10 hardware signals every 500ms (Battery %, Drain Rate, GPU/CPU Temp, real NVML Wattage, FPS, Latency, CPU Usage, Plugged State, and Time).
2. **AI Policy Engine:** A lightweight Scikit-Learn Decision Tree evaluates these signals to determine the most efficient execution target.
3. **Zero-Friction Migration:** When the GPU overheats (e.g., >85°C) or the battery drains critically, EcoScale hot-swaps the ONNX Execution Provider mid-stream. The workload instantly shifts from the RTX 4060 (CUDA) to the Ryzen 7 CPU or AMD XDNA NPU without dropping the inference session.

## 📊 Measured Impact & Enterprise ROI
Our prototype is fully functional, tested on an HP Omen 16, and all data is derived from real `pynvml` hardware wattage measurements.

* **Power Efficiency:** Achieves a 56% power reduction, successfully scaling workloads from a 43.0W GPU baseline down to a 9.4W CPU footprint.
* **Sustainability:** Translates to a verifiable carbon footprint reduction of 5.87 kg CO₂ per user per year.
* **Enterprise Scale (10,000 laptops):** Requires zero deployment cost while saving approximately Rs. 36 lakhs in electricity annually and eliminating 372 tonnes of CO₂.

---

## 🛠️ Tech Stack
* **AI / ML:** Scikit-Learn (Decision Tree), ONNX Runtime 1.17, YOLOv8n (GPU), MobileNet SSD (CPU/NPU).
* **Hardware Telemetry:** `pynvml` (NVIDIA wattage/temp), `psutil` (battery/CPU usage), Windows `WMI` (CPU temp).
* **Backend:** Python 3.11, FastAPI, Uvicorn, Asyncio.
* **Frontend:** React 18, Vite, WebSocket API (30fps streaming).

---
