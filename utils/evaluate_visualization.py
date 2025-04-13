import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# 配置参数
CONF_THRESH = 0.25  # 置信度阈值
IOU_THRESH = 0.45   # IOU阈值
SAVE_DIR = "/localdata/kyuak/Rune-Detection/models/test"  # 结果保存目录

def load_yolo_keypoints(label_path, img_width, img_height):
    """加载YOLO格式的关键点标注"""
    if not os.path.exists(label_path):
        return None
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # 只取第一个有效对象
    for line in lines:
        data = list(map(float, line.strip().split()))
        if len(data) < 9:  # 至少需要class_id + bbox + 1个点
            continue
            
        class_id = int(data[0])
        # 提取关键点 (格式: class_id, x_center, y_center, w, h, x1,y1,v1, x2,y2,v2,...)
        kpts = np.array(data[5:]).reshape(-1, 3)
        # 转换为像素坐标
        kpts[:, 0] *= img_width   # x
        kpts[:, 1] *= img_height  # y
        return {
            'class_id': class_id,
            'keypoints': kpts,
            'bbox': np.array([data[1]*img_width, data[2]*img_height, 
                             data[3]*img_width, data[4]*img_height])  # x_center,y_center,w,h
        }
    return None

def visualize_and_save(img, true_obj, pred_objs, save_path):
    """可视化预测结果和真实标注并保存"""
    # 复制图像避免修改原图
    vis_img = img.copy()
    
    # 绘制真实标注 (绿色)
    if true_obj:
        # 绘制4个关键点
        for i, (x, y, v) in enumerate(true_obj['keypoints']):
            if v > 0:  # 只绘制可见点
                # cv2.circle(vis_img, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(vis_img, f"{i+1}", (int(x)+10, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制真实bbox
        x, y, w, h = true_obj['bbox']
        # cv2.rectangle(vis_img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
        cv2.putText(vis_img, f"True (Class {true_obj['class_id']})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 绘制预测结果 (红色)
    for i, pred in enumerate(pred_objs):
        # 绘制4个关键点
        for j, (x, y, v) in enumerate(pred['keypoints']):
            if v > 0:
                cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.putText(vis_img, f"{j+1}", (int(x)+10, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制预测bbox
        kpts = pred['keypoints']
        x_min, y_min = kpts[:, 0].min(), kpts[:, 1].min()
        x_max, y_max = kpts[:, 0].max(), kpts[:, 1].max()
        # cv2.rectangle(vis_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        cv2.putText(vis_img, f"Pred {i+1} (Class {pred['class_id']}, conf={pred['conf']:.2f})", 
                    (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis_img)

def run_visualization(model, test_img_dir, test_label_dir):
    """运行可视化流程"""
    # 创建结果目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 遍历测试集
    img_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for img_name in tqdm(img_files, desc="Processing images"):
        img_path = os.path.join(test_img_dir, img_name)
        label_path = os.path.join(test_label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # 加载图像和真实标注
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        height, width = img.shape[:2]
        true_obj = load_yolo_keypoints(label_path, width, height)
        
        # 模型预测
        results = model.predict(img_path, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
        pred_objs = []
        
        for result in results:
            for i, box in enumerate(result.boxes):
                if box.conf < CONF_THRESH:
                    continue
                
                # 获取关键点
                kpt = result.keypoints[i]
                kpt_coords = []
                for x, y, conf in kpt.data[0].tolist():
                    kpt_coords.append([x, y, 1])  # 假设所有点可见
                
                pred_objs.append({
                    'class_id': int(box.cls),
                    'keypoints': np.array(kpt_coords),
                    'conf': float(box.conf)
                })
        
        # 保存可视化结果
        save_path = os.path.join(SAVE_DIR, img_name)
        visualize_and_save(img, true_obj, pred_objs, save_path)

if __name__ == "__main__":
    # 加载训练好的模型
    model = YOLO("/localdata/kyuak/Rune-Detection/models/tianda1/weights/best.pt")
    
    # 测试集路径
    test_img_dir = "/localdata/kyuak/Rune-Detection/dataset/split_data/split3/test/images"
    test_label_dir = "/localdata/kyuak/Rune-Detection/dataset/split_data/split3/test/labels"
    
    # 运行可视化
    run_visualization(model, test_img_dir, test_label_dir)
    print(f"\nVisualization completed. Results saved to {SAVE_DIR}")