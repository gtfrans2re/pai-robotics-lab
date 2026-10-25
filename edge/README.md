# Edge AI — OpenVINO on CPU / Arc iGPU / NPU

Run inference across all three accelerators on the Meteor Lake target.

## Prerequisites

The `edge` + `pyenv` setup modules (OpenVINO, the iGPU compute runtime, and the
Intel NPU `.deb` driver):

```bash
./setup/setup_robotics.sh pyenv edge
source ~/pai-robotics-venv/bin/activate
```

## Check devices

```bash
python check_devices.py
# Expected: ['CPU', 'GPU', 'NPU']
```

If `NPU` is missing, see the NPU note in the top-level README — the confined
`intel-npu-driver` snap is invisible to pip-installed OpenVINO; the `.deb`
driver fixes it (the `edge` module installs the `.deb` path).

## Benchmark YOLOv11n

```bash
python benchmark_yolo_openvino.py
```

Exports YOLOv11n to an OpenVINO IR (once), then times inference on each device
and prints latency + throughput. A reproducible baseline for comparing CPU vs
iGPU vs NPU on constrained hardware.

## Notes

- First run downloads `yolo11n.pt` and writes `yolo11n_openvino_model/` (gitignored).
- Default export is FP32. For an INT8 NPU-friendly model, export with quantization
  (`YOLO(...).export(format="openvino", int8=True, data=...)`) — needs a calibration set.
