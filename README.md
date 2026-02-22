# EcoScale — Setup Guide (HP Omen 16 · AMD Ryzen 7 + RTX 4060)

## What Actually Happens in Real Mode

| Event | What the code does |
|---|---|
| Laptop plugged in | `psutil` detects AC → `set_profile("GPU")` called → ONNX session switches to `CUDAExecutionProvider` → YOLOv8n inference runs on RTX 4060 CUDA cores |
| Cable unplugged | `psutil` detects battery → `set_profile("NPU")` called → ONNX session switches to `CPUExecutionProvider` (or VitisAI if installed) → MobileNet SSD runs lightweight inference |
| Power reading | `pynvml` reads real milliwatts from NVML driver and divides by 1000 for actual watts |

---

## Step 1 — Python Environment

```bash
# Create a clean environment
python -m venv ecoscale_env
ecoscale_env\Scripts\activate        # Windows

pip install -r requirements.txt
```

## Step 2 — CUDA Setup (for RTX 4060)

onnxruntime-gpu requires CUDA 12.x. Verify:
```bash
nvidia-smi                           # Should show your RTX 4060
nvcc --version                       # Should show CUDA 12.x
```

If CUDA is not installed:
- Download CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads
- Then: `pip install onnxruntime-gpu`

## Step 3 — Download AI Models

```bash
python download_models.py
```

This downloads:
- `models/yolov8n.onnx` — ~12MB, used on GPU (RTX 4060)
- `models/mobilenet_ssd.onnx` — ~28MB, used on CPU/NPU (battery mode)

## Step 4 — Optional: AMD XDNA NPU Support

Only available on Ryzen 7040+ series with XDNA architecture.
Check if your Ryzen 7 has NPU:
```bash
# In Device Manager look for "NPU Compute Accelerators"
# or check AMD's Ryzen AI compatible processor list
```

If your chip supports it:
1. Download Ryzen AI Software: https://ryzenai.docs.amd.com/
2. Install the SDK
3. `pip install onnxruntime-vitisai`
4. Place `vaip_config.json` (from the SDK) in the project folder

Without NPU: battery mode uses optimized CPU threads instead — still more efficient than GPU mode.

## Step 5 — Start Backend

```bash
python ecoscale_backend.py
```

You should see:
```
  GPU (CUDA)    : ✓ NVIDIA GeForce RTX 4060 Laptop GPU
  NPU (VitisAI) : ✗ Ryzen AI SDK needed   (or ✓ if installed)
  Power (NVML)  : ✓ Real wattage
  YOLOv8n       : ✓
  MobileNet SSD : ✓
```

## Step 6 — Frontend

Open `EcoScale.jsx` in the Claude artifact viewer, or:

```bash
# In a new terminal
npm create vite@latest ecoscale-ui -- --template react
cd ecoscale-ui
cp ../EcoScale.jsx src/App.jsx
npm install
npm run dev
```

---

## Real Behavior to Expect

**When plugged in:**
- YOLOv8n loads on RTX 4060
- NVML reports actual GPU wattage (~30-55W depending on scene)
- ~55-60 FPS, ~15-25ms inference
- Bounding boxes drawn in orange

**When you unplug the cable:**
- `psutil` detects the event in <500ms
- ONNX session immediately switches providers
- MobileNet SSD activates on CPU/NPU
- FPS drops to ~25-35, power drops to ~8-15W
- Bounding boxes drawn in green

The video feed stays continuous with no black frame — the switch happens mid-stream.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDAExecutionProvider not available` | Install CUDA 12.x + `pip install onnxruntime-gpu` |
| `No module named pynvml` | `pip install nvidia-ml-py` |
| Wattage shows "Estimated" | pynvml not installed or NVML driver issue — run `nvidia-smi` to check |
| Camera not opening | Change `cv2.VideoCapture(0)` index to 1 or 2 in backend |
| Model file missing | Run `python download_models.py` |
