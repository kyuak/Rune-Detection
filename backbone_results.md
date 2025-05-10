# 1. yolo11n-pose

### classification:

==== Time Statistics ====
Mean Inference Time: 0.0323s
Std Inference Time:  0.1362s
Min Inference Time:  0.0101s
Max Inference Time:  1.5264s

==== Class-wise Metrics ====
 RedInactive: Precision=1.0000, Recall=1.0000, F1=1.0000, Support=64
   RedActive: Precision=1.0000, Recall=0.9600, F1=0.9796, Support=25
BlueInactive: Precision=1.0000, Recall=1.0000, F1=1.0000, Support=59
  BlueActive: Precision=1.0000, Recall=0.8000, F1=0.8889, Support=10

==== Overall Metrics ====
Micro Precision: 1.0000
Micro Recall:    0.9810
Micro F1-Score:  0.9904

==== Confusion Matrix ====
             RedInactive RedActive BlueInactive BlueActive
RedInactive        64       0       0       0
RedActive           0      24       0       0
BlueInactive        0       0      59       0
BlueActive          0       0       0       8

### pose_accuracy:

==== Keypoint Evaluation Metrics ====
Mean OKS: 0.9606 ± 0.0504

AP@0.95 per class:
 RedInactive: 0.8281
BlueInactive: 0.6949

AR@0.05 per class:
 RedInactive: 1.0000
BlueInactive: 1.0000

# 2. yolo11s-pose

### classification:

==== Time Statistics ====
Mean Inference Time: 0.0320s
Std Inference Time:  0.1356s
Min Inference Time:  0.0100s
Max Inference Time:  1.5196s

==== Class-wise Metrics ====
 RedInactive: Precision=1.0000, Recall=1.0000, F1=1.0000, Support=64
   RedActive: Precision=1.0000, Recall=0.9600, F1=0.9796, Support=25
BlueInactive: Precision=1.0000, Recall=1.0000, F1=1.0000, Support=59
  BlueActive: Precision=1.0000, Recall=0.9000, F1=0.9474, Support=10

==== Overall Metrics ====
Micro Precision: 1.0000
Micro Recall:    0.9873
Micro F1-Score:  0.9936

==== Confusion Matrix ====
             RedInactive RedActive BlueInactive BlueActive
RedInactive        64       0       0       0
RedActive           0      24       0       0
BlueInactive        0       0      59       0
BlueActive          0       0       0       9

### pose_accuracy:

==== Keypoint Evaluation Metrics ====
Mean OKS: 0.9616 ± 0.0605

AP@0.95 per class:
 RedInactive: 0.8438
BlueInactive: 0.7966

AR@0.05 per class:
 RedInactive: 1.0000
BlueInactive: 1.0000

# 3. yolov8n-pose

### classification:

==== Time Statistics ====
Mean Inference Time: 0.0298s
Std Inference Time:  0.1292s
Min Inference Time:  0.0086s
Max Inference Time:  1.4467s

==== Class-wise Metrics ====
 RedInactive: Precision=1.0000, Recall=1.0000, F1=1.0000, Support=64
   RedActive: Precision=1.0000, Recall=1.0000, F1=1.0000, Support=25
BlueInactive: Precision=1.0000, Recall=1.0000, F1=1.0000, Support=59
  BlueActive: Precision=1.0000, Recall=0.8000, F1=0.8889, Support=10

==== Overall Metrics ====
Micro Precision: 1.0000
Micro Recall:    0.9873
Micro F1-Score:  0.9936

==== Confusion Matrix ====
             RedInactive RedActive BlueInactive BlueActive
RedInactive        64       0       0       0
RedActive           0      25       0       0
BlueInactive        0       0      59       0
BlueActive          0       0       0       8

### pose_accuracy:

==== Keypoint Evaluation Metrics ====
Mean OKS: 0.9385 ± 0.0565

AP@0.95 per class:
 RedInactive: 0.5938
BlueInactive: 0.4746

AR@0.05 per class:
 RedInactive: 1.0000
BlueInactive: 1.0000

# 4. yolov8s-pose

### classification:

==== Time Statistics ====
Mean Inference Time: 0.0307s
Std Inference Time:  0.1366s
Min Inference Time:  0.0088s
Max Inference Time:  1.5302s

==== Class-wise Metrics ====
 RedInactive: Precision=0.9275, Recall=1.0000, F1=0.9624, Support=64
   RedActive: Precision=0.8621, Recall=1.0000, F1=0.9259, Support=25
BlueInactive: Precision=1.0000, Recall=1.0000, F1=1.0000, Support=59
  BlueActive: Precision=0.9091, Recall=1.0000, F1=0.9524, Support=10

==== Overall Metrics ====
Micro Precision: 0.9405
Micro Recall:    1.0000
Micro F1-Score:  0.9693

==== Confusion Matrix ====
             RedInactive RedActive BlueInactive BlueActive
RedInactive        64       0       0       0
RedActive           0      25       0       0
BlueInactive        0       0      59       0
BlueActive          0       0       0      10

### pose_accuracy:

==== Keypoint Evaluation Metrics ====
Mean OKS: 0.9568 ± 0.0508

AP@0.95 per class:
 RedInactive: 0.7969
BlueInactive: 0.7458

AR@0.05 per class:
 RedInactive: 1.0000
BlueInactive: 1.0000
