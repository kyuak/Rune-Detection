import os
import cv2
import numpy as np

def convert_to_yolo_pose(original_label, image_width, image_height):
    """
    将4点标注转换为YOLOv8 Pose格式（带可见性标志）
    输入格式: [class_id, x1,y1, x2,y2, x3,y3, x4,y4] (归一化坐标)
    输出格式: [class_id, x_center, y_center, width, height, 
              x1,y1,1, x2,y2,1, x3,y3,1, x4,y4,1] (归一化坐标)
    """
    # 提取类别和点
    class_id = int(original_label[0])
    points = np.array(original_label[1:]).reshape(-1, 2)  # 转换为(N,2)数组
    
    # 转换为像素坐标
    pixel_points = points * np.array([image_width, image_height])
    
    # 计算最小外接矩形 (返回: (x,y,w,h))
    rect = cv2.boundingRect(pixel_points.astype(np.int32))
    x, y, w, h = rect
    
    # 计算归一化的边界框中心点和宽高
    x_center = (x + w/2) / image_width
    y_center = (y + h/2) / image_height
    width = w / image_width
    height = h / image_height
    
    # 构建YOLO Pose格式（在每个x,y后添加可见性标志1）
    yolo_pose_label = [class_id, x_center, y_center, width, height]
    for i in range(4):
        yolo_pose_label.extend([original_label[1+i*2], original_label[2+i*2], 1])  # 添加x,y,1
    
    return yolo_pose_label

def process_folder(images_folder, labels_folder, output_folder):
    """
    处理整个文件夹，转换所有标签
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for label_file in os.listdir(labels_folder):
        if not label_file.endswith('.txt'):
            continue
            
        # 获取对应的图片路径
        image_file = os.path.splitext(label_file)[0] + '.jpg'  # 假设图片是jpg格式
        image_path = os.path.join(images_folder, image_file)
        
        # 读取图片获取尺寸
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
            else:
                print(f"Warning: Could not read image {image_path}, using default size 640x640")
                w, h = 640, 640
        else:
            print(f"Warning: Image not found {image_path}, using default size 640x640")
            w, h = 640, 640
        
        # 读取原始标签
        label_path = os.path.join(labels_folder, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 处理每个标注行
        new_lines = []
        for line in lines:
            original_label = list(map(float, line.strip().split()))
            if len(original_label) != 9:  # 检查格式是否正确
                print(f"Warning: Invalid label format in {label_file}: {line}")
                continue
            
            # 转换为YOLO Pose格式（带可见性标志）
            yolo_pose_label = convert_to_yolo_pose(original_label, w, h)
            new_lines.append(' '.join(map(str, yolo_pose_label)) + '\n')
        
        # 写入新标签
        output_path = os.path.join(output_folder, label_file)
        with open(output_path, 'w') as f:
            f.writelines(new_lines)

if __name__ == "__main__":
    images_folder = "/home/ykw/Rune-Detection/dataset/raw_data/rune_2024_v2.0/images/train"
    labels_folder = "/home/ykw/Rune-Detection/dataset/raw_data/rune_2024_v2.0/labels/train"
    output_folder = "/home/ykw/Rune-Detection/dataset/raw_data/rune_2024_v2.0/new_labels/"
    
    process_folder(images_folder, labels_folder, output_folder)
    print(f"Conversion completed. Converted labels saved to {output_folder}")