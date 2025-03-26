import torch
from ultralytics import YOLO

# 加载预训练模型
model_name = "yolo11n-pose"
model = YOLO("/localdata/kyuak/Rune-Detection/models/" + model_name + ".pt")

# 训练配置
results = model.train(
    data="/localdata/kyuak/Rune-Detection/dataset/v11n.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device="0",
    workers=4,
    project="/localdata/kyuak/Rune-Detection/models",
    name=model_name,
    exist_ok=True,
    augment=True,
)

default_args = model.trainer.args
print("所有训练参数:\n", vars(default_args))