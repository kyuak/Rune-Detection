from ultralytics import YOLO
 
# 加载一个模型，路径为 YOLO 模型的 .pt 文件
model = YOLO("/localdata/kyuak/Rune-Detection/models/test1/weights/best.pt")
 
# 导出模型，设置多种参数
model.export(
    format="onnx",      # 导出格式为 ONNX
    imgsz=(640, 640),   # 设置输入图像的尺寸
    keras=False,        # 不导出为 Keras 格式
    optimize=False,     # 不进行优化 False, 移动设备优化的参数，用于在导出为TorchScript 格式时进行模型优化
    half=False,         # 不启用 FP16 量化
    int8=False,         # 不启用 INT8 量化
    dynamic=False,      # 不启用动态输入尺寸
    simplify=True,      # 简化 ONNX 模型
    opset=None,         # 使用最新的 opset 版本
    workspace=4.0,      # 为 TensorRT 优化设置最大工作区大小（GiB）
    batch=1,            # 指定批处理大小
    device="cpu",        # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"
    nms=True,             # 包含NMS
    task="keypoint"       # 明确指定关键点任务
)

# from ultralytics import YOLO

# model = YOLO('/localdata/kyuak/RM2025-DatasetUtils/models/rune_blender/rune_blender+lab_v0.1/weights/best.pt')
# model.export(
#     format="openvino",
#     dynamic=False,
#     half=False,  # 强制使用 FP32 避免精度损失
#     simplify=True,  # 简化模型结构
#     nms=False,  # 禁用内置 NMS
#     agnostic_nms=False,  # 禁用类别无关 NMS
#     imgsz=640,  # 输入图像大小
# )