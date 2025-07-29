ROS2 Object Detection with YOLOv11
This repository contains ROS2 packages for real-time object detection using the cutting-edge YOLOv11 model. The project aims to provide a robust and efficient solution for integrating advanced object detection capabilities into ROS2-based robotic applications.

Features
ROS2 integration for YOLOv11 object detection.

Support for exporting YOLOv11 models to ONNX format for optimized inference.

Modular structure for easy extension and customization.

Installation
To set up and run this project, please follow these steps:

Clone the Repository:

git clone [https://github.com/JornDeruyck/ros2_object_detection.git](https://github.com/JornDeruyck/ros2_object_detection.git)
cd ros2_object_detection

Create a Python Virtual Environment (Recommended):

python3 -m venv venv
source venv/bin/activate

Install Dependencies:
Install the ultralytics package, which is required for YOLOv11 model operations.

pip install ultralytics

You may also need other ROS2 dependencies depending on your specific setup (e.g., rosdep install --from-paths src --ignore-src -r -y).

Build the ROS2 Workspace:

colcon build

Source the Workspace:

source install/setup.bash

Generating YOLOv11 ONNX Model
The pre-trained YOLOv11 model weights (.pt files) are not included directly in this repository due to their size and specific licensing. Instead, they are downloaded on demand by the ultralytics library, and then converted to the ONNX format for efficient deployment.

To generate the yolov11l.onnx model file in your models/ directory:

Ensure your Python environment is activated (if you created one during installation).

source venv/bin/activate # If you used a virtual environment

Run the generation script:

python scripts/generate_yolov11.py

This script will:

Automatically download the yolo11l.pt (PyTorch) weights if they are not already cached on your system.

Convert the yolo11l.pt model to yolov11l.onnx.

Save the yolov11l.onnx file into your models/ directory.

Note: An active internet connection is required for the initial download of the .pt weights. The models/ directory is ignored by Git, so the generated .onnx file will not be committed to this repository.

Usage
(This section is a placeholder. You can expand it with instructions on how to run your ROS2 nodes, launch files, and use the object detection capabilities once the model is generated.)

Example:

# Example command to run your ROS2 detection node
ros2 run ros2_object_detection detection_node
