import os
import numpy as np

# 定义图像和标注文件夹路径
images_folder = "/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/images/val"
labels_folder = "/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/dataset/labels/val"

# 关键点数量
num_keypoints = 4  # 根据你的数据集调整

# 检查标注文件格式
def check_label_file(label_path, num_keypoints):
    with open(label_path, "r") as file:
        lines = file.readlines()

    errors = []
    for line_num, line in enumerate(lines, start=1):
        values = line.strip().split()
        # 检查值的数量是否正确
        if len(values) != 1 + 4 + num_keypoints * 3:  # class_id + bbox + keypoints
            errors.append(f"Line {line_num}: Invalid number of values ({len(values)}). Expected {1 + 4 + num_keypoints * 3}.")
            continue
        
        # 检查关键点是否在 [0, 1] 范围内
        keypoints = list(map(float, values[5:]))
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if not (0 <= x <= 1 and 0 <= y <= 1 and v in [0, 1]):
                errors.append(f"Line {line_num}: Invalid keypoint values ({x}, {y}, {v}).")
    
    return errors

# 检查图像和标注文件的对应关系
def check_image_label_pairs(images_folder, labels_folder):
    image_files = [f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))]
    label_files = [f for f in os.listdir(labels_folder) if f.endswith(".txt")]

    # 检查是否有图像文件没有对应的标注文件
    missing_labels = []
    for img_file in image_files:
        label_file = img_file[:-4] + ".txt"
        if label_file not in label_files:
            missing_labels.append(img_file)

    # 检查是否有标注文件没有对应的图像文件
    missing_images = []
    for label_file in label_files:
        img_file = label_file[:-4] + ".jpg"
        if img_file not in image_files:
            missing_images.append(label_file)

    return missing_labels, missing_images

# 主函数
def main():
    # 检查图像和标注文件的对应关系
    missing_labels, missing_images = check_image_label_pairs(images_folder, labels_folder)
    if missing_labels:
        print("以下图像文件没有对应的标注文件：")
        for img_file in missing_labels:
            print(f"- {img_file}")
    if missing_images:
        print("以下标注文件没有对应的图像文件：")
        for label_file in missing_images:
            print(f"- {label_file}")

    # 检查标注文件格式
    print("\n检查标注文件格式...")
    for label_file in os.listdir(labels_folder):
        if label_file.endswith(".txt"):
            label_path = os.path.join(labels_folder, label_file)
            errors = check_label_file(label_path, num_keypoints)
            if errors:
                print(f"标注文件 {label_file} 有以下错误：")
                for error in errors:
                    print(f"  {error}")
            # else:
                # print(f"标注文件 {label_file} 格式正确。")

if __name__ == "__main__":
    main()