import sys
sys.path.insert(0, '/localdata/kyuak/Rune-Detection')
import ultralytics
from ultralytics import YOLO
# print("🚀 正在使用的Ultralytics路径:", ultralytics.__file__)

model = YOLO('/localdata/kyuak/Rune-Detection/dataset/yolo11-pose-v1.0.yaml')
print("✅ 成功加载模型!")

results = model.train(
    data="/localdata/kyuak/Rune-Detection/dataset/v11n.yaml",
    epochs=150,
    batch=128,
    imgsz=640,
    amp=False,
    optimizer='AdamW',
    lr0=0.01,
    # box=4.0, cls=0.2, dfl=1.5, kobj=3.0,
    # box=7.5, cls=0.5, dfl=1.5, kobj=1.0
    device="1,2,3,4",
    workers=8,
    project="/localdata/kyuak/Rune-Detection/models",
    name="test1.5",
    exist_ok=True,
    augment=False,
)

default_args = model.trainer.args
print("所有训练参数:\n", vars(default_args))