# Rune-Detection 🔍⚙️

This repository provides the codebase and experimental setup for our real-time rune detection and classification framework, designed for RoboMaster competition scenarios.

## Overview

This project builds upon the [Ultralytics YOLOv8/11](https://github.com/ultralytics/ultralytics) framework, with custom modifications and additions to support keypoint detection and pose estimation. Our final model integrates attention mechanisms like **ECA** and **CBAM**, and is optimized using **Distribution Focal Loss (DFL)** for improved localization accuracy.

## Repository Structure

├── dataset/         # Contains training/testing data and YOLO-style model configs

├── docs/            # Documentation files

├── examples/        # Usage and testing examples

├── tests/           # Unit and functional tests

├── ultralytics/     # Modified YOLOv11 codebase with custom modules

│   └── nn/modules/  # Contains ECA and CBAM modules integrated into the backbone

├── utils/           # Experimental and visualization scripts

├── README.md        # Project overview
