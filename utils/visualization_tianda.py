import os
import cv2
import numpy as np
from time import sleep

images_folder = "/home/ykw/Rune-Detection/dataset/raw_data/rune_2024_v2.0/images/train"
labels_folder = "/home/ykw/Rune-Detection/dataset/raw_data/rune_2024_v2.0/labels/train"

# 确保文件夹存在
assert os.path.exists(images_folder), f"Images folder not found: {images_folder}"
assert os.path.exists(labels_folder), f"Labels folder not found: {labels_folder}"

# 获取所有图片文件
image_files = sorted([f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for img_file in image_files:
    # 构建对应的标签文件路径
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(labels_folder, label_file)
    
    # 读取图片
    img_path = os.path.join(images_folder, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image {img_path}")
        continue
    
    h, w = image.shape[:2]
    
    # 检查标签文件是否存在
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found for {img_file}")
        cv2.imshow("Image", image)
        cv2.waitKey(2000)
        continue
    
    # 读取标签数据
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # 处理每个标注
    # only deal with labels with more than 1 line
    if len(lines) < 2:
        continue

    for line in lines:
        data = list(map(float, line.strip().split()))
        if len(data) != 9:  # 检查数据格式是否正确
            print(f"Warning: Invalid data format in {label_file}: {line}")
            continue
        
        class_id = int(data[0])

        if class_id == 0:
            color = (0, 255, 0)
        elif class_id == 1:
            color = (255, 255, 0)

        points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
        
        # 转换归一化坐标到像素坐标
        pixel_points = (points * np.array([w, h])).astype(np.int32)
        
        # 绘制多边形
        cv2.polylines(image, [pixel_points], isClosed=True, color=color, thickness=2)
        
        # 绘制点（可选）
        for i, (x, y) in enumerate(pixel_points):
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.putText(image, f"{i}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 显示类别（可选）
        # cv2.putText(image, f"Class: {class_id}", pixel_points[0] + (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # 显示图片2000ms
    cv2.imshow("Annotation Viewer", image)
    key = cv2.waitKey(200)  # 显示2000ms
    
    # 如果按下ESC键，提前退出
    if key == 27:
        break

cv2.destroyAllWindows()
print("Annotation visualization completed.")