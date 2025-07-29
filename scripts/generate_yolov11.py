import os
import shutil
from ultralytics import YOLO

def generate_yolov11_onnx():
    """
    Loads the YOLOv11l model, exports it to ONNX format with a specific input size,
    and moves the ONNX model to the specified /models directory.
    """
    print("Starting YOLOv11l ONNX model generation...")

    # Define the target models directory relative to the script's location
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True) # Ensure the models directory exists

    # Define the desired final output filename
    output_onnx_filename = 'yolov11l.onnx'
    final_output_path = os.path.join(models_dir, output_onnx_filename)

    # Define the desired input image size for the ONNX model to match DeepStream config
    # This must match the H and W in infer-dims=C;H;W in pgie_yolov11_config.txt
    target_img_size = 1280 

    try:
        # Load a pretrained YOLOv11l model.
        print(f"Loading YOLOv11l model from 'yolo11l.pt'. This may download the model if not found locally.")
        model = YOLO('yolo11l.pt')

        # Export the model to ONNX format.
        # Crucially, set imgsz to match the infer-dims in the DeepStream config.
        print(f"Exporting YOLOv11l model to ONNX format with imgsz={target_img_size} and saving to: {final_output_path}")
        export_results = model.export(format='onnx', imgsz=target_img_size, name='yolov11l', opset=12) 
        
        exported_file_path = export_results 

        print(f"Model initially exported to: {exported_file_path}")

        # Move the exported ONNX file to the desired models directory
        print(f"Moving exported model to: {final_output_path}")
        shutil.move(exported_file_path, final_output_path)

        print(f"YOLOv11l ONNX model successfully exported and moved to: {final_output_path}")

    except Exception as e:
        print(f"An error occurred during model generation: {e}")
        print("Please ensure you have the 'ultralytics' package installed (`pip install ultralytics`)")
        print("and that you have an active internet connection for initial model download.")

if __name__ == "__main__":
    generate_yolov11_onnx()
