# Auto-download and export YOLO11n ONNX
# Run once: python setup_model.py
import subprocess, os, shutil
MODEL_NAME="yolo11n"
OUTPUT_DIR="models"
OUTPUT_FILE=os.path.join(OUTPUT_DIR,f"{MODEL_NAME}.onnx")
os.makedirs(OUTPUT_DIR, exist_ok=True)
subprocess.check_call(["pip","install","ultralytics","onnx","onnxruntime"])
subprocess.check_call(["yolo","export",f"model={MODEL_NAME}.pt","format=onnx","opset=12"])
if os.path.exists(f"{MODEL_NAME}.onnx"):
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
    shutil.move(f"{MODEL_NAME}.onnx",OUTPUT_FILE)
    print(f"Model ready at {OUTPUT_FILE}")
else:
    print("Export finished but ONNX file not found.")
