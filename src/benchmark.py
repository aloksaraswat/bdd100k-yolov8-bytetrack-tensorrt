"""Benchmark FP32 / FP16 / INT8: accuracy + latency, then patch the README.

Defensive by design: any single measurement that fails (e.g. validating a
fixed-batch engine) is reported as "-" rather than crashing the whole run, so
we always end up with a results table and a patched README.
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch


def first_val_image(data_yaml):
    import yaml
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    vd = Path(cfg["path"]) / cfg["val"]
    for p in sorted(vd.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            return str(p)
    raise RuntimeError(f"no val images under {vd}")


def measure_latency(model, img, imgsz, device, runs, warmup):
    for _ in range(warmup):
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    inf = []
    for _ in range(runs):
        r = model.predict(img, imgsz=imgsz, device=device, verbose=False)
        inf.append(r[0].speed["inference"])
    torch.cuda.synchronize()
    mean = statistics.mean(inf)
    mem = torch.cuda.max_memory_allocated() / 1e6
    return round(mean, 2), round(1000.0 / mean, 1), round(mem, 1)


def measure_map(model, data_yaml, imgsz, device):
    # batch=1 matches the static batch of the exported engines
    m = model.val(data=data_yaml, imgsz=imgsz, device=device, batch=1,
                  verbose=False, plots=False)
    return round(float(m.box.map50), 4), round(float(m.box.map), 4)


def bench_variant(name, weights, data_yaml, img, imgsz, device, runs, warmup):
    from ultralytics import YOLO
    print(f"\n=== {name} ({weights}) ===")
    row = {"variant": name, "map50": "-", "map5095": "-",
           "infer_ms": "-", "infer_fps": "-", "mem": "-"}
    try:
        model = YOLO(weights)
    except Exception as e:
        print("  load failed:", e)
        return row
    try:
        ms, fps, mem = measure_latency(model, img, imgsz, device, runs, warmup)
        row.update(infer_ms=ms, infer_fps=fps, mem=mem)
        print(f"  latency {ms} ms  ({fps} FPS)")
    except Exception as e:
        print("  latency failed:", e)
    try:
        m50, m = measure_map(model, data_yaml, imgsz, device)
        row.update(map50=m50, map5095=m)
        print(f"  mAP50 {m50}  mAP50-95 {m}")
    except Exception as e:
        print("  mAP failed:", e)
    return row


def measure_track_fps(weights, video, imgsz, device):
    from ultralytics import YOLO
    model = YOLO(weights)
    n = 0
    t0 = time.perf_counter()
    for _ in model.track(source=video, tracker="bytetrack.yaml", imgsz=imgsz,
                         device=device, persist=True, stream=True, save=False,
                         verbose=False):
        n += 1
    return round(n / (time.perf_counter() - t0), 1), n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("GPU:", gpu)

    try:
        img = first_val_image(args.data)
    except Exception as e:
        print("no val image, using a blank frame:", e)
        from PIL import Image
        img = "/tmp/blank.jpg"
        Image.new("RGB", (1280, 720)).save(img)

    rows = [
        bench_variant("PyTorch FP32", args.fp32, args.data, img, args.imgsz, args.device, args.runs, args.warmup),
        bench_variant("TensorRT FP16", args.fp16, args.data, img, args.imgsz, args.device, args.runs, args.warmup),
        bench_variant("TensorRT INT8", args.int8, args.data, img, args.imgsz, args.device, args.runs, args.warmup),
    ]

    tfps, tframes = "-", "-"
    try:
        tfps, tframes = measure_track_fps(args.int8, args.video, args.imgsz, args.device)
    except Exception as e:
        print("track fps failed:", e)

    lines = [
        f"_Measured on **{gpu}**, {datetime.now():%Y-%m-%d}. "
        f"Single-image inference at 640x640, averaged over many runs._", "",
        "| Variant | mAP@50 | mAP@50-95 | Inference | FPS | Peak GPU mem |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['map50']} | {r['map5095']} | "
                     f"{r['infer_ms']} ms | **{r['infer_fps']}** | {r['mem']} MB |")
    lines += ["", f"End-to-end detect + ByteTrack on the demo clip: "
              f"**{tfps} FPS** ({tframes} frames, INT8 engine)."]
    table = "\n".join(lines)

    Path("results.json").write_text(json.dumps(
        {"gpu": gpu, "rows": rows, "track_fps": tfps, "track_frames": tframes}, indent=2))

    rd = Path(args.readme)
    t = rd.read_text()
    s, e = "<!-- BENCH:START -->", "<!-- BENCH:END -->"
    block = f"{s}\n{table}\n{e}"
    if s in t and e in t:
        rd.write_text(t[:t.index(s)] + block + t[t.index(e) + len(e):])
    else:
        rd.write_text(t.rstrip() + "\n\n## Results\n\n" + block + "\n")

    print("\n" + table)
    print("\nbenchmark complete - README patched")


if __name__ == "__main__":
    main()
