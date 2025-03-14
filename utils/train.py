import torch
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("/localdata/kyuak/RM2025-DatasetUtils/yolo11n-pose.pt")

# 训练配置
results = model.train(
    data="/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/v11n.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device="0",
    workers=8,
    project="/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/models",
    name="test1",
    exist_ok=True,
    augment=False,  # 禁用数据增强
)