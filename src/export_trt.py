"""Export the trained detector to TensorRT engines (FP16 and INT8).

Two engines are built:

  * FP16 - half precision, the usual edge default. Near-lossless accuracy,
    roughly 2x faster than FP32.
  * INT8 - 8-bit integer. Fastest and smallest, but it needs a calibration
    pass over real images so TensorRT can pick per-tensor scales. We calibrate
    on a slice of the BDD val set. INT8 trades a little mAP for latency; the
    benchmark reports exactly how much so the tradeoff is visible, not hidden.

A TensorRT engine is built for the *specific* GPU it is created on, so this
must run on the same T4 you benchmark on.

    python src/export_trt.py --weights weights/best.pt --data datasets/bdd2/bdd2.yaml
"""
import argparse
import shutil
from pathlib import Path


def export_one(weights, imgsz, device, half=False, int8=False, data=None,
               fraction=0.3):
    from ultralytics import YOLO

    model = YOLO(weights)
    kwargs = dict(format="engine", imgsz=imgsz, device=device, workspace=4,
                  verbose=False)
    if half:
        kwargs["half"] = True
    if int8:
        kwargs["int8"] = True
        kwargs["data"] = data
        kwargs["fraction"] = fraction
    path = model.export(**kwargs)
    return Path(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="bdd2.yaml, for INT8 calibration")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--calib-fraction", type=float, default=0.3)
    args = ap.parse_args()

    weights = Path(args.weights)
    out_dir = weights.parent

    print("Exporting FP16 engine ...")
    fp16 = export_one(args.weights, args.imgsz, args.device, half=True)
    fp16_dst = out_dir / "best_fp16.engine"
    shutil.move(str(fp16), str(fp16_dst))
    print(f"  -> {fp16_dst}")

    print("Exporting INT8 engine (calibrating on BDD val) ...")
    int8 = export_one(args.weights, args.imgsz, args.device, int8=True,
                      data=args.data, fraction=args.calib_fraction)
    int8_dst = out_dir / "best_int8.engine"
    shutil.move(str(int8), str(int8_dst))
    print(f"  -> {int8_dst}")


if __name__ == "__main__":
    main()
