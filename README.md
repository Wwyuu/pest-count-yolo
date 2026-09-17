# pest-count-yolo

Stored-grain pest **counting** with a dual-head YOLOv8:

- **Detect**: YOLOv8s with an added **P2** (stride-4) small-object scale  
- **Density map generation module**: shallow head on P2/P3 → non-negative map \(D\)  
- **Decode**: exclusive connected-component routing — integrate dense blobs, count NMS boxes elsewhere  

One forward pass yields boxes and \(D\); the reported count is the routed count (not whole-image \(\sum D\)).

## Layout

```
pest-count-yolo/
  configs/yolov8s-p2.yaml   # P2 multi-scale Detect
  configs/data.example.yaml # Ultralytics data yaml template
  yolo_den/
    dual_yolo.py            # DualYoloDen + density module
    decode.py               # blob routing (mode=blob)
    gaussian.py             # points → Gaussian density GT
  train_detector.py         # stage-1: train Detect (P2 or plain v8s)
  train.py                  # stage-2: joint Detect + density module
  eval.py                   # evaluate den / blob / nms_fixed
  requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Prepare data under `./data` (or set `COUNT_DATA`):

```
data/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt   # each line: x y  (pixel center)  [class optional]
  yolo/pest.yaml                  # Ultralytics detect yaml (see configs/data.example.yaml)
  yolo/labels/{train,val,test}/   # pseudo-boxes for Detect (cx cy w h normalized), if used
```

Environment variables (optional):

| Variable | Default | Meaning |
|---|---|---|
| `COUNT_DATA` | `./data` | dataset root |
| `RUNS_DIR` | `./runs` | checkpoints / logs |
| `DET_CKPT` | `runs/YOLOv8-P2-SUM/train/weights/best.pt` | stage-1 Detect weights for joint train |
| `YOLO_DATA_YAML` | `$COUNT_DATA/yolo/pest.yaml` | Ultralytics data yaml |
| `SWANLAB_API_KEY` | (empty) | optional experiment logging |
| `CUDA_VISIBLE_DEVICES` | (system) | GPU id |

## Train

**1. Detector (with P2)**

```bash
python train_detector.py --which yolov8p2 --epochs 80 --batch 8 --out runs/YOLOv8-P2-SUM
```

Ablation without P2: `--which yolov8s`.

**2. Joint dual-head** (init from stage-1 `best.pt`; freeze Detect 3 epochs, then joint train)

```bash
python train.py --ckpt runs/YOLOv8-P2-SUM/train/weights/best.pt --head p2p3 --epochs 25 --batch 4 --out runs/YOLO-DenHead-SUM
```

Loss: \(L_{\mathrm{YOLO}} + 10\,\mathrm{MSE} + 0.1\,|\mathrm{sum}| + 0.05\,L_n\) (`L_n` = normalized density L1).

## Eval

```bash
python eval.py --ckpt runs/YOLO-DenHead-SUM/best.pt --split test --out runs/eval-yolo-den --tag YOLOv8-P2
```

Primary metric for the paper method: **blob** (CC routing). `mae_den` is whole-map \(\sum D\) (baseline-style), not the main score.

## License

Code is provided for research use. Please cite the accompanying paper when available.

## Acknowledgement

Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
