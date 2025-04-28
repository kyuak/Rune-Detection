from ultralytics import YOLO

model = YOLO('C:\Users\ykw\Desktop\Rune-Detection\ultralytics\cfg\models\11\yolo11-pose-v1.0.yaml')  # 你的改过的架构
model.train(data='/path/to/your/rune.yaml', epochs=100, imgsz=640, optimizer='AdamW', lr0=0.01)

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