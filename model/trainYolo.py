from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
import shutil
from ultralytics import YOLO
import os


run_name = "Glands_Finder_Final_old_dataset_150"
base_path = f"runs/train/{run_name}"

model = YOLO("yolo11n.pt")

model.train(
    data='../preprocessedData_v1/LearnSet/dataAugumented.yaml',
    epochs=150,
    imgsz=640,
    project="runs/train",
    name=run_name,
    exist_ok=True
)

source_path = f"runs/train/{run_name}/weights/best.pt"
destination_path = f"saved_models/{run_name}_best.pt"
os.makedirs("saved_models", exist_ok=True)
shutil.copy(source_path, destination_path)
