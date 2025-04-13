import torch
from ultralytics import YOLO

# 加载预训练模型
model_name = "yolo11n-pose"
model = YOLO("/localdata/kyuak/Rune-Detection/models/" + model_name + ".pt")

# 训练配置
results = model.train(
    data="/localdata/kyuak/Rune-Detection/dataset/v11n.yaml",
    epochs=150,
    batch=256,
    imgsz=640,
    device="4,5,6,7",
    workers=8,
    project="/localdata/kyuak/Rune-Detection/models",
    name="tianda1",
    exist_ok=True,
    augment=False,
)

default_args = model.trainer.args
print("所有训练参数:\n", vars(default_args))