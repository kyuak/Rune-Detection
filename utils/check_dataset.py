import os

# 定义图像和标注文件夹路径
images_folder = '/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/tune_data/images'
labels_folder = '/localdata/kyuak/RM2025-DatasetUtils/models/rune_test/tune_data/labels'

# 检查文件夹是否存在
if not os.path.exists(images_folder):
    print(f"Images folder {images_folder} does not exist.")
    exit()

if not os.path.exists(labels_folder):
    print(f"Labels folder {labels_folder} does not exist.")
    exit()

# 遍历 images 文件夹中的所有 .jpg 文件
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg"):
        # 获取对应的 .txt 文件路径
        txt_filename = filename[:-4] + ".txt"  # 将 .jpg 替换为 .txt
        txt_path = os.path.join(labels_folder, txt_filename)

        # 如果对应的 .txt 文件不存在
        if not os.path.exists(txt_path):
            # 删除没有对应 .txt 文件的 .jpg 文件
            img_path = os.path.join(images_folder, filename)
            os.remove(img_path)
            print(f"Deleted {img_path} (no corresponding .txt file)")