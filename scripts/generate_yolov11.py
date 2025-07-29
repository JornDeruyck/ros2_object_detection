import os
import shutil # Import shutil for moving files
from ultralytics import YOLO

def generate_yolov11_onnx():
    """
    Loads the YOLOv11l model, exports it to ONNX format,
    and moves the ONNX model to the specified /models directory.
    """
    print("Starting YOLOv11l ONNX model generation...")

    # Define the target models directory relative to the script's location
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True) # Ensure the models directory exists

    # Define the desired final output filename
    output_onnx_filename = 'yolov11l.onnx'
    final_output_path = os.path.join(models_dir, output_onnx_filename)

    try:
        # Load a pretrained YOLOv11l model.
        # This will automatically download the .pt weights if they are not already present.
        print(f"Loading YOLOv11l model from 'yolo11l.pt'. This may download the model if not found locally.")
        model = YOLO('yolo11l.pt')

        # Export the model to ONNX format.
        # The 'export' method will save the file to a default location (e.g., runs/export/yolo11l.onnx).
        # We specify 'yolo11l' as the name for the exported file, which will be used in the default path.
        print(f"Exporting YOLOv11l model to ONNX format...")
        export_results = model.export(format='onnx', imgsz=1024, name='yolov11l') 
        
        # The export_results object contains the path to the exported file.
        # We need to extract the actual path of the generated ONNX file.
        # Ultralytics typically saves to 'runs/export/name_of_model.onnx'
        # The export_results object directly returns the path of the exported file.
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
