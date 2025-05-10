import os
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from collections import defaultdict
import cv2

# 配置参数
TEST_IMAGES_DIR = "/localdata/kyuak/Rune-Detection/dataset/split_data/split3/test/images"
TEST_LABELS_DIR = "/localdata/kyuak/Rune-Detection/dataset/split_data/split3/test/labels"
MODEL_PATH = "/localdata/kyuak/Rune-Detection/models/test1.5/weights/best.pt"
# CLASS_NAMES = ['RedInactive', 'RedActive', 'BlueInactive', 'BlueActive']
CLASS_NAMES = ['Inactive', 'Active']
KEYPOINT_NAMES = ['point1', 'point2', 'point3', 'point4']
CONF_THRESH = 0.8
IOU_THRESH = 0.6
OKS_SIGMA = 0.1
AP_THRESHOLD = 0.95
AR_THRESHOLD = 0.05

def load_yolo_keypoints(label_path, img_width, img_height):
    """读取YOLO格式的关键点标签（只提取类别0或2的第一个对象）"""
    if not os.path.exists(label_path):
        return None
    
    with open(label_path) as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        cls_id = int(parts[0])
        if cls_id not in [0, 2]:  # 只处理类别0和2
            continue
            
        # 解析关键点 (x,y,visibility)
        kpts = []
        for i in range(5, len(parts), 3):
            x = float(parts[i]) * img_width
            y = float(parts[i+1]) * img_height
            vis = int(parts[i+2])
            kpts.append([x, y, vis])
        
        return {
            'class_id': cls_id,
            'keypoints': np.array(kpts),
            'scale': max(float(parts[3]) * img_width, float(parts[4]) * img_height) * OKS_SIGMA
        }
    return None

def calculate_oks(true_kpts, pred_kpts, scale):
    """计算OKS（关键点顺序一一对应）"""
    vis_mask = true_kpts[:, 2] > 0
    if vis_mask.sum() == 0:
        return 0.0
    
    # 按顺序直接计算对应点距离（假设顺序一致）
    d = np.linalg.norm(true_kpts[vis_mask, :2] - pred_kpts[vis_mask, :2], axis=1)
    return np.exp(-(d**2) / (2 * scale**2)).mean()

def evaluate_keypoints(model, test_img_dir, test_label_dir):
    stats = {
        'oks_scores': [],
        'ap_per_class': defaultdict(list),
        'ar_per_class': defaultdict(list)
    }
    
    for img_name in os.listdir(test_img_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        img_path = os.path.join(test_img_dir, img_name)
        label_path = os.path.join(test_label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # 加载真实标注（只取第一个类别0或2的对象）
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        true_obj = load_yolo_keypoints(label_path, width, height)
        if true_obj is None:
            continue
            
        # 模型预测（过滤非0/2类别）
        results = model.predict(img_path, device='7', conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
        pred_objs = []

        for result in results:
            class_id = None
            confidence = 0.0
            for i, box in enumerate(result.boxes):
                if int(box.cls) in [0, 2] and box.conf >= CONF_THRESH:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    kpt = result.keypoints[i]
                    kpt_coords = []
                    for j, (x, y) in enumerate(kpt.xy[0].tolist()):
                        kpt_coords.append([x, y, 1])
                    kpt_coords = np.array(kpt_coords)
                    break
            if class_id is None:
                continue

            pred_objs.append({
                'class_id': class_id,
                'keypoints': kpt_coords,
                'conf': confidence
            })
        
        # 匹配同类别预测（取置信度最高的）
        matched_pred = None
        for pred_obj in pred_objs:
            if pred_obj['class_id'] == true_obj['class_id']:
                if matched_pred is None or pred_obj['conf'] > matched_pred['conf']:
                    matched_pred = pred_obj
        
        # 计算指标
        if matched_pred is not None:
            oks = calculate_oks(true_obj['keypoints'], matched_pred['keypoints'], true_obj['scale'])
            cls_name = CLASS_NAMES[true_obj['class_id']]
            stats['oks_scores'].append(oks)
            stats['ap_per_class'][cls_name].append(float(oks > AP_THRESHOLD))
            stats['ar_per_class'][cls_name].append(float(oks > AR_THRESHOLD))
    
    # 计算结果（与原输出格式一致）
    AP_name = 'AP@' + str(AP_THRESHOLD)
    AR_name = 'AR@' + str(AR_THRESHOLD)
    metrics = {
        'OKS_mean': np.mean(stats['oks_scores']) if stats['oks_scores'] else 0,
        'OKS_std': np.std(stats['oks_scores']) if stats['oks_scores'] else 0,
        AP_name: {k: np.mean(v) if v else 0 for k, v in stats['ap_per_class'].items()},
        AR_name: {k: np.mean(v) if v else 0 for k, v in stats['ar_per_class'].items()}
    }
    
    print("\n==== Keypoint Evaluation Metrics ====")
    print(f"Mean OKS: {metrics['OKS_mean']:.4f} ± {metrics['OKS_std']:.4f}")
    print("\n" + AP_name + " per class:")
    for cls, score in metrics[AP_name].items():
        print(f"{cls:>12}: {score:.4f}")
    print("\n" + AR_name + " per class:")
    for cls, score in metrics[AR_name].items():
        print(f"{cls:>12}: {score:.4f}")

if __name__ == "__main__":
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    evaluate_keypoints(model, TEST_IMAGES_DIR, TEST_LABELS_DIR)