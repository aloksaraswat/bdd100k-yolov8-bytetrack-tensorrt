"""Fine-tune YOLOv8 on the person/vehicle BDD100K subset.

We start from COCO-pretrained weights. COCO already knows `person` and the
vehicle classes, so fine-tuning is mostly domain adaptation to BDD's dashcam
viewpoint, night scenes and weather. That is why a relatively short schedule
already gives a usable detector.

    python src/train.py --data datasets/bdd2/bdd2.yaml --epochs 20
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="path to bdd2.yaml")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="base weights (yolov8n/s/m.pt)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="bdd2_y8s")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="resume an interrupted run from its last.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = args.model
    if args.resume:
        # resume picks up optimizer state from the run's last checkpoint
        weights = f"{args.project}/{args.name}/weights/last.pt"

    model = YOLO(weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        amp=True,            # mixed precision: ~2x throughput on a T4
        cos_lr=True,         # cosine LR decay, plays well with a short schedule
        plots=True,          # save PR / confusion-matrix plots for the README
        resume=args.resume,
        exist_ok=True,
    )

    best = f"{args.project}/{args.name}/weights/best.pt"
    print(f"\nBest weights: {best}")


if __name__ == "__main__":
    main()
