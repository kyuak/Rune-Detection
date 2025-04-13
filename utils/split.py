import os
import random
import shutil

# 定义路径
dataset_path = '/localdata/kyuak/Rune-Detection/dataset/raw_data/rune_tianda_v1.0'
images_folder = dataset_path + '/images'
labels_folder = dataset_path + '/labels'
output_folder = "/localdata/kyuak/Rune-Detection/dataset/split_data/split3"  # 输出文件夹

# 创建输出文件夹
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, split, "labels"), exist_ok=True)

# 获取所有图像文件名
image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
random.shuffle(image_files)  # 随机打乱

# 划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 计算划分点
train_split = int(len(image_files) * train_ratio)
val_split = train_split + int(len(image_files) * val_ratio)

# 划分数据集
for i, filename in enumerate(image_files):
    if i < train_split:
        split = "train"
    elif i < val_split:
        split = "val"
    else:
        split = "test"

    # 复制图像文件
    shutil.copy(
        os.path.join(images_folder, filename),
        os.path.join(output_folder, split, "images", filename)
    )
    # 复制标注文件
    label_filename = filename[:-4] + ".txt"
    shutil.copy(
        os.path.join(labels_folder, label_filename),
        os.path.join(output_folder, split, "labels", label_filename)
    )

print("Dataset split completed!")