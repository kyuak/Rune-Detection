import cv2
import os
from ultralytics import YOLO

# 定义路径
model_path = "/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/models/test1/weights/best.pt"  # 模型路径
image_path = "/localdata/kyuak/RM2025-DatasetUtils/rune/samples/1255.jpg"  # 输入图片路径
output_path = "/localdata/kyuak/RM2025-DatasetUtils/rune/samples/predicted_1255.jpg"  # 输出图片路径

# 类别名称
class_names = {
    0: "RedInactive",
    1: "RedActive",
    2: "BlueInactive",
    3: "BlueActive"
}

# 加载训练好的模型
model = YOLO(model_path)

# 预测图片
results = model.predict(image_path)

# 加载图片
image = cv2.imread(image_path)
height, width, _ = image.shape

# 处理预测结果
for result in results:
    # 获取检测框信息
    boxes = result.boxes
    for box in boxes:
        # 提取检测框坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 获取类别 ID 和置信度
        class_id = int(box.cls)
        confidence = float(box.conf)
        
        if class_id == 2 or class_id == 3:
            color = (0, 0, 255)  # 蓝色类别用红色标注
        else:
            color = (255, 0, 0)  # 红色类别用蓝色标注

        # 绘制类别名称和置信度
        class_name = class_names.get(class_id, "Unknown")
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 获取关键点信息（如果模型支持关键点检测）
    if hasattr(result, "keypoints"):
        keypoints = result.keypoints
        for kpt in keypoints:
            # 提取关键点坐标
            kpt_coords = kpt.xy[0].tolist()
            for i, (x, y) in enumerate(kpt_coords):
                # 绘制关键点
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
                cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 保存绘制后的图片
cv2.imwrite(output_path, image)
print(f"Predicted image saved to {output_path}")