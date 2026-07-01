# Real-time person and vehicle detection with tracking on edge GPU

A small end-to-end pipeline. It fine-tunes YOLOv8 on BDD100K for two classes
(person and vehicle), runs ByteTrack on top for multi-object tracking, exports
the detector to TensorRT in FP16 and INT8, and benchmarks all three on a T4. A
T4 is a fair stand-in for the kind of GPU you'd sit next to a camera.

The focus is the deployment path rather than a leaderboard score: how fast the
model runs once it's quantized, and how much accuracy that costs.

## Results

<!-- BENCH:START -->
_Measured on **Tesla T4**, 2026-07-01. Single-image inference at 640x640, averaged over many runs._

| Variant | mAP@50 | mAP@50-95 | Inference | FPS | Peak GPU mem |
|---|---|---|---|---|---|
| PyTorch FP32 | 0.6365 | 0.351 | 6.0 ms | **166.5** | 68.1 MB |
| TensorRT FP16 | 0.6386 | 0.3491 | 3.2 ms | **312.7** | 55.7 MB |
| TensorRT INT8 | 0.629 | 0.3344 | 3.45 ms | **290.1** | 55.7 MB |

End-to-end detect + ByteTrack on the demo clip: **128.5 FPS** (647 frames, INT8 engine).
<!-- BENCH:END -->

## Notes on the main choices

I went with YOLOv8s because it hits a good speed and accuracy point on a T4. The
anchor-free head keeps NMS cheap and it exports to TensorRT cleanly. yolov8n is
an easy swap for more speed, yolov8m if there's headroom to spare.

I collapsed BDD's ten categories down to two, person and vehicle. For a "person
near a moving machine" safety problem that is the signal that matters, and it
keeps both classes well populated. The mapping lives in src/prepare_data.py.

For tracking I used ByteTrack. It keeps the low-confidence detections around
during association, which is what lets it hold an ID through a brief occlusion
instead of dropping the track and starting a new one. I kept the bytetrack.yaml
that ships with Ultralytics so the tracker stays reproducible.

INT8 is where most of the edge latency savings come from. It needs a calibration
pass over real images, for which I used a slice of the BDD validation set, and
it costs a little accuracy. The benchmark reports FP32, FP16 and INT8 next to
each other so that cost is visible instead of hidden.

## Dataset

BDD100K, taken in YOLO format from the public Kaggle mirror
a7madmostafa/bdd100k-yolo. src/prepare_data.py reads it, remaps the categories
to person and vehicle by name, splits into train and val, and writes a clean
YOLO dataset with a bdd2.yaml.

| class | BDD categories |
|---|---|
| person | pedestrian, rider |
| vehicle | car, truck, bus, train, motorcycle, bicycle |
| dropped | traffic light, traffic sign |

It trains on a 10k-image slice at an 80/20 split, which is enough to fine-tune
from COCO weights and get a usable detector in one session. Point it at the full
train set and raise the epoch count to scale up; nothing else changes.

## How to reproduce

Open notebooks/colab_runner.ipynb in Colab on a free T4, set the runtime to GPU
and run all. It installs the dependencies, builds the dataset, trains, runs the
tracker, exports both engines, benchmarks, and writes the numbers back here.

To run the steps directly on any CUDA machine:

```bash
pip install -r requirements.txt
kaggle datasets download -d a7madmostafa/bdd100k-yolo -p kaggle_bdd --unzip
python src/prepare_data.py --src kaggle_bdd --out datasets/bdd2
python src/train.py --data datasets/bdd2/bdd2.yaml --epochs 20
python src/export_trt.py --weights weights/best.pt --data datasets/bdd2/bdd2.yaml
python src/track.py --weights weights/best.pt --source clip.mp4
python src/benchmark.py --fp32 weights/best.pt --fp16 weights/best_fp16.engine --int8 weights/best_int8.engine --data datasets/bdd2/bdd2.yaml --video clip.mp4
```

## Repo layout

```
src/prepare_data.py   build the 2-class YOLO dataset from BDD100K
src/train.py          fine-tune YOLOv8
src/track.py          ByteTrack plus an annotated clip and GIF
src/export_trt.py     build the FP16 and INT8 TensorRT engines
src/benchmark.py      accuracy and latency, writes the table above
weights/best.pt       the trained detector
```

## Limitations

TensorRT engines are tied to the GPU they were built on, so a committed engine
came from a T4 and needs rebuilding for other hardware.

The numbers are from a single T4 and a 10k-image fine-tune. More data and a
longer schedule would raise the accuracy; the latency picture stays the same.

Tracking quality is shown qualitatively with the demo clip. A proper MOTA/IDF1
score would need the separate BDD100K MOT split, which I kept out of scope here
rather than report a number I hadn't actually measured.
