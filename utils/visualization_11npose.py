import cv2
import os
import numpy as np

# 定义图像和标注文件夹路径
images_folder = "/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/images/train"
labels_folder = "/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/labels/train"
output_folder = "/localdata/kyuak/RM2025-DatasetUtils/rune/samples"  # 保存图像的输出文件夹

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 类别名称
class_names = {
    0: "RedInactive",
    1: "RedActive",
    2: "BlueInactive",
    3: "BlueActive"
}

# 计数器，用于保存前三张图像
save_count = 0

# 遍历文件夹中的所有图像文件
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 获取图像文件的完整路径
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # 加载对应的标注文件
        label_path = os.path.join(labels_folder, filename[:-4] + ".txt")
        if not os.path.exists(label_path):
            print(f"Label file not found for {filename}")
            continue

        with open(label_path, "r") as file:
            lines = file.readlines()

        # 处理每一行标注
        for line in lines:
            # 删除行末尾的换行符并分割数值
            values = line.strip().split()

            # 提取类别、检测框和关键点信息
            class_id = int(values[0])
            bbox = list(map(float, values[1:5]))  # 检测框 (center_x, center_y, w, h)
            keypoints = list(map(float, values[5:]))  # 关键点 (x1, y1, v1, x2, y2, v2, ...)

            # 将归一化坐标转换为实际像素坐标
            center_x = int(bbox[0] * width)
            center_y = int(bbox[1] * height)
            w = int(bbox[2] * width)
            h = int(bbox[3] * height)

            # 绘制检测框
            x1 = int(center_x - w / 2)
            y1 = int(center_y - h / 2)
            x2 = int(center_x + w / 2)
            y2 = int(center_y + h / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制类别名称
            class_name = class_names.get(class_id, "Unknown")
            if class_id == 2 or class_id == 3:
                color = (0, 0, 255)  # 蓝色类别用红色标注
            else:
                color = (255, 0, 0)  # 红色类别用蓝色标注
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 绘制关键点
            for i in range(0, len(keypoints), 3):
                x = int(keypoints[i] * width)
                y = int(keypoints[i + 1] * height)
                visibility = keypoints[i + 2]

                if visibility > 0:  # 只绘制可见的关键点
                    cv2.circle(image, (x, y), 5, color, -1)
                    cv2.putText(image, str(i // 3), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存前三张图像
        if save_count < 3:
            output_path = os.path.join(output_folder, f"visualization_{save_count + 1}.jpg")
            cv2.imwrite(output_path, image)
            print(f"Saved visualization to {output_path}")
            save_count += 1

        # 如果已经保存了三张图像，退出循环
        if save_count >= 3:
            break