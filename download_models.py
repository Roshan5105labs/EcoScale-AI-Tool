"""
EcoScale — Model Setup
Downloads real ONNX models used for GPU and NPU/CPU inference.

"""

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    # YOLOv8n — Nano variant, optimized for speed, runs on CUDA (RTX 4060)
    "yolov8n.onnx": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "note": "Will be exported to ONNX via ultralytics",
        "use_ultralytics": True,
    },
    # MobileNetV2 SSD — lightweight, for CPU/NPU mode
    "mobilenet_ssd.onnx": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd-mobilenet-v1-12.onnx",
        "note": "MobileNet SSD — direct ONNX download",
        "use_ultralytics": False,
    },
}


def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)


def export_yolov8_to_onnx():
    out_path = os.path.join(MODELS_DIR, "yolov8n.onnx")
    if os.path.exists(out_path):
        print(f"  [✓] yolov8n.onnx already exists, skipping.")
        return out_path

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  Installing ultralytics for model export...")
        os.system(f"{sys.executable} -m pip install ultralytics -q")
        from ultralytics import YOLO

    print("  Downloading YOLOv8n weights and exporting to ONNX...")
    model = YOLO("yolov8n.pt") 
    model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)

    # ultralytics saves next to the .pt file — move it
    pt_dir = os.path.expanduser("~/.config/Ultralytics") if os.name != "nt" else "."
    onnx_src = "yolov8n.onnx"
    if os.path.exists(onnx_src):
        os.rename(onnx_src, out_path)
    print(f"\n  [✓] Saved to {out_path}")
    return out_path


def download_mobilenet():
    out_path = os.path.join(MODELS_DIR, "mobilenet_ssd.onnx")
    if os.path.exists(out_path):
        print(f"  [✓] mobilenet_ssd.onnx already exists, skipping.")
        return out_path

    url = MODELS["mobilenet_ssd.onnx"]["url"]
    print(f"  Downloading MobileNet SSD ONNX...")
    try:
        urllib.request.urlretrieve(url, out_path, reporthook=progress_bar)
        print(f"\n  [✓] Saved to {out_path}")
    except Exception as e:
        print(f"\n  [✗] Failed: {e}")
        print("      Falling back to torchvision export...")
        _export_mobilenet_via_torch(out_path)
    return out_path


def _export_mobilenet_via_torch(out_path):
    """Fallback: export MobileNetV2 from torchvision to ONNX."""
    try:
        import torch
        import torchvision
        print("  Exporting MobileNetV2 via torchvision...")
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model, dummy, out_path,
            opset_version=12,
            input_names=["images"],
            output_names=["output"],
        )
        print(f"  [✓] Exported MobileNetV2 to {out_path}")
    except Exception as e:
        print(f"  [✗] torchvision export also failed: {e}")
        print("  The backend will fall back to OpenCV DNN mode.")


if __name__ == "__main__":
    print("=" * 55)
    print("  EcoScale — Model Setup")
    print("=" * 55)

    print("\n[1/2] YOLOv8n (GPU mode — RTX 4060 via CUDA)")
    export_yolov8_to_onnx()

    print("\n[2/2] MobileNet SSD (Battery/NPU mode — CPU EP)")
    download_mobilenet()

    print("\n" + "=" * 55)
    print("  Done. Run: python ecoscale_backend.py")
    print("=" * 55)
