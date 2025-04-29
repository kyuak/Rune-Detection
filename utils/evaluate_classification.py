import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# 配置参数
TEST_IMAGES_DIR = "/localdata/kyuak/Rune-Detection/dataset/split_data/split3/test/images"
TEST_LABELS_DIR = "/localdata/kyuak/Rune-Detection/dataset/split_data/split3/test/labels"
MODEL_PATH = "/localdata/kyuak/Rune-Detection/models/test2/weights/best.pt"
# CLASS_NAMES = ['RedInactive', 'RedActive', 'BlueInactive', 'BlueActive']
CLASS_NAMES = ['Inactive', 'Active']
CONF_THRESH = 0.063  # 根据F1曲线选择的最佳阈值
DEBUG = False

def load_true_labels(label_path):
    """从YOLO格式标签文件中读取真实类别"""
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return [int(line.strip().split()[0]) for line in lines if line.strip()]

def evaluate_model(model, test_img_dir, test_label_dir):
    # 初始化统计变量
    confusion_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    stats = {
        'true_positives': defaultdict(int),
        'false_positives': defaultdict(int),
        'false_negatives': defaultdict(int),
        'support': defaultdict(int)
    }

    times = []
    # 遍历测试集
    for img_name in os.listdir(test_img_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(test_img_dir, img_name)
        label_path = os.path.join(test_label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # 获取真实标签
        true_classes = load_true_labels(label_path)
        if not true_classes:
            continue  # 跳过没有标签的图像

        # 记录每个类别的样本数
        for cls in true_classes:
            stats['support'][cls] += 1

        # 模型预测
        start_time = time.perf_counter()
        results = model.predict(img_path, device='6', conf=CONF_THRESH, verbose=False)
        # results = model.predict(img_path, conf=CONF_THRESH, verbose=False)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        pred_classes = []
        if results[0].boxes:
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist()

        # 重置匹配状态
        matched_pred_indices = set()

        # 匹配真实标签和预测
        for true_cls in true_classes:
            matched = False
            for i, pred_cls in enumerate(pred_classes):
                if i not in matched_pred_indices and pred_cls == true_cls:
                    confusion_matrix[true_cls][pred_cls] += 1
                    stats['true_positives'][true_cls] += 1
                    matched_pred_indices.add(i)
                    matched = True
                    break
            if not matched:
                if DEBUG:
                    print(f"漏检: 图像 {img_name} 未检测到 {CLASS_NAMES[true_cls]}")
                    print(f"真实标签: {true_classes}")
                    print(f"预测标签: {pred_classes}")
                stats['false_negatives'][true_cls] += 1

        # 统计未匹配的预测作为FP
        for i, pred_cls in enumerate(pred_classes):
            if i not in matched_pred_indices:
                stats['false_positives'][pred_cls] += 1
                if DEBUG:
                    print(f"误检: 图像 {img_name} 错误预测为 {CLASS_NAMES[pred_cls]}")
                    print(f"真实标签: {true_classes}")
                    print(f"预测标签: {pred_classes}")

    # 计算指标
    def safe_divide(a, b):
        return a / b if b != 0 else 0.0

    # 类级别指标
    class_metrics = {}
    for cls in range(len(CLASS_NAMES)):
        tp = stats['true_positives'][cls]
        fp = stats['false_positives'][cls]
        fn = stats['false_negatives'][cls]
        
        if DEBUG:
            print(f"\n==== {CLASS_NAMES[cls]} ====")
            print(f"TP: {tp}, FP: {fp}, FN: {fn}, Total: {stats['support'][cls]}")

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': stats['support'][cls]
        }

    # 总体指标
    total_tp = sum(stats['true_positives'].values())
    total_fp = sum(stats['false_positives'].values())
    total_fn = sum(stats['false_negatives'].values())
    
    micro_precision = safe_divide(total_tp, total_tp + total_fp)
    micro_recall = safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    # 打印结果
    print("\n==== Time Statistics ====")
    print(f"Mean Inference Time: {mean_time:.4f}s")
    print(f"Std Inference Time:  {std_time:.4f}s")
    print(f"Min Inference Time:  {min_time:.4f}s")
    print(f"Max Inference Time:  {max_time:.4f}s")

    print("\n==== Class-wise Metrics ====")
    for cls, metrics in class_metrics.items():
        print(f"{CLASS_NAMES[cls]:>12}: "
              f"Precision={metrics['precision']:.4f}, "
              f"Recall={metrics['recall']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"Support={metrics['support']}")

    print("\n==== Overall Metrics ====")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall:    {micro_recall:.4f}")
    print(f"Micro F1-Score:  {micro_f1:.4f}")

    print("\n==== Confusion Matrix ====")
    print_confusion_matrix(confusion_matrix, CLASS_NAMES)

def print_confusion_matrix(matrix, class_names):
    """打印格式化的混淆矩阵"""
    max_len = max(len(name) for name in class_names)
    header = " " * (max_len + 1) + " ".join(f"{name:>8}" for name in class_names)
    print(header)
    
    for i, row in enumerate(matrix):
        print(f"{class_names[i]:<{max_len}}", end=" ")
        for val in row:
            print(f"{val:8}", end="")
        print()

if __name__ == "__main__":
    # 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # 执行评估
    evaluate_model(model, TEST_IMAGES_DIR, TEST_LABELS_DIR)