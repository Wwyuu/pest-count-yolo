# pest-count-yolo

双头 YOLOv8 储粮害虫计数：Detect（含 P2）+ 密度图生成模块（P2/P3）+ 连通域分流解码（密区积分、疏区数框）。

## 结构

```
configs/yolov8s-p2.yaml
yolo_den/          # DualYoloDen、密度模块、高斯真值、blob 解码
train_detector.py  # 先训检测
train.py           # 联合训练
eval.py            # 评测
requirements.txt
```

## 使用

```bash
pip install -r requirements.txt

# 数据根目录默认 ./data，也可用环境变量 COUNT_DATA
python train_detector.py --which yolov8p2 --epochs 80 --batch 8 --out runs/det
python train.py --ckpt runs/det/train/weights/best.pt --head p2p3 --epochs 25 --batch 4 --out runs/den
python eval.py --ckpt runs/den/best.pt --split test --out runs/eval
```

依赖 Ultralytics YOLOv8。
