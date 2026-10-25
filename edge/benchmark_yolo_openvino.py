"""Export YOLOv11n to OpenVINO and benchmark inference latency on each device.

Runs the same model on every OpenVINO device (CPU / GPU / NPU) and reports
mean latency and throughput — a clean "edge AI on constrained hardware" baseline.

Usage:  python benchmark_yolo_openvino.py
"""
import os
import time

import numpy as np
import openvino as ov
from ultralytics import YOLO

IMG = 640          # export/inference resolution
N = 50             # timed iterations per device
WARMUP = 5
EXPORT_DIR = "yolo11n_openvino_model"


def export_once():
    if not os.path.isdir(EXPORT_DIR):
        print("Exporting YOLOv11n -> OpenVINO IR (first run only) ...")
        YOLO("yolo11n.pt").export(format="openvino", imgsz=IMG)
    xml = next(
        os.path.join(EXPORT_DIR, f) for f in os.listdir(EXPORT_DIR) if f.endswith(".xml")
    )
    return xml


def main():
    xml = export_once()
    core = ov.Core()
    model = core.read_model(xml)
    x = np.random.rand(1, 3, IMG, IMG).astype(np.float32)

    print(f"\nDevices: {core.available_devices}")
    print(f"Model:   {xml}   input {x.shape}   iters {N}\n")
    print(f"{'device':6s} {'latency':>12s} {'throughput':>14s}")
    print("-" * 34)

    for dev in core.available_devices:
        try:
            compiled = core.compile_model(model, dev)
            for _ in range(WARMUP):
                compiled([x])
            t0 = time.perf_counter()
            for _ in range(N):
                compiled([x])
            ms = (time.perf_counter() - t0) / N * 1000.0
            print(f"{dev:6s} {ms:9.2f} ms {1000.0 / ms:10.1f} FPS")
        except Exception as e:  # noqa: BLE001
            print(f"{dev:6s}  skipped: {e}")


if __name__ == "__main__":
    main()
