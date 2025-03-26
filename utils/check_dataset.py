import os

# 定义图像和标注文件夹路径
dataset_path = '/localdata/kyuak/Rune-Detection/dataset/raw_data/rune_combine_v1.0'

images_folder = dataset_path + '/images'
labels_folder = dataset_path + '/labels'

delete = True

# 检查文件夹是否存在
if not os.path.exists(images_folder):
    print(f"Images folder {images_folder} does not exist.")
    exit()

if not os.path.exists(labels_folder):
    print(f"Labels folder {labels_folder} does not exist.")
    exit()

# Supported image extensions (add more if needed)
image_extensions = ('.jpg', '.jpeg', '.png') 

# --- Check 1: Images without labels ---
for img_filename in os.listdir(images_folder):
    if img_filename.lower().endswith(image_extensions):
        # Remove image extension and add .txt
        base_name = os.path.splitext(img_filename)[0]  # Splits 'image.jpg' to ('image', '.jpg')
        txt_filename = base_name + ".txt"
        txt_path = os.path.join(labels_folder, txt_filename)
        if not os.path.exists(txt_path):
            img_path = os.path.join(images_folder, img_filename)
            if delete:
                os.remove(img_path)
            print(f"Check labels: Deleted {img_path} (no corresponding .txt file)")

# --- Check 2: Labels without images ---
for txt_filename in os.listdir(labels_folder):
    if txt_filename.endswith(".txt"):
        base_name = os.path.splitext(txt_filename)[0]  # Splits 'label.txt' to ('label', '.txt')
        
        # Check for any matching image extension (e.g., .jpg, .png)
        img_exists = False
        for ext in image_extensions:
            img_filename = base_name + ext
            img_path = os.path.join(images_folder, img_filename)
            if os.path.exists(img_path):
                img_exists = True
                break
        if not img_exists:
            txt_path = os.path.join(labels_folder, txt_filename)
            if delete:
                os.remove(txt_path)
            print(f"Check images: Deleted {txt_path} (no corresponding .jpg/.png image)")