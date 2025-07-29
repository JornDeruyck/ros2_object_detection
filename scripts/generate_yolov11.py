import os
from ultralytics import YOLO

def generate_yolov11_onnx():
    """
    Loads the YOLOv11l model, exports it to ONNX format,
    and saves the ONNX model to the specified /models directory.
    """
    print("Starting YOLOv11l ONNX model generation...")

    # Define the path to the models directory relative to the script's location
    # Assuming this script is in 'your_repo/scripts/' and models are in 'your_repo/models/'
    # os.path.dirname(__file__) gets the directory of the current script (scripts/)
    # '..' moves up one level to the parent directory (your_repo/)
    # 'models' then points to the models directory (your_repo/models/)
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Ensure the models directory exists. If it doesn't, create it.
    os.makedirs(models_dir, exist_ok=True) 

    # Define the full path for the output ONNX model, including the filename
    output_onnx_filename = 'yolov11l.onnx'
    output_onnx_path = os.path.join(models_dir, output_onnx_filename)

    try:
        # Load a pretrained YOLOv11l model.
        # This will automatically download the .pt weights if they are not already present.
        print(f"Loading YOLOv11l model from 'yolo11l.pt'. This may download the model if not found locally.")
        model = YOLO('yolo11l.pt')

        # Export the model to ONNX format.
        # We now explicitly pass the full output path using the 'filename' argument.
        print(f"Exporting YOLOv11l model to ONNX format and saving to: {output_onnx_path}")
        # The 'export' method will save the file to the path specified in 'filename'.
        # imgsz=1024 is used as YOLOv11 models often use this input size for ONNX export.
        model.export(format='onnx', imgsz=1024, filename=output_onnx_path) 

        print(f"YOLOv11l ONNX model successfully exported to: {output_onnx_path}")

    except Exception as e:
        print(f"An error occurred during model generation: {e}")
        print("Please ensure you have the 'ultralytics' package installed (`pip install ultralytics`)")
        print("and that you have an active internet connection for initial model download.")

if __name__ == "__main__":
    generate_yolov11_onnx()
