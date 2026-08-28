from ultralytics import YOLO
from cv.albumentations_hook import inject_custom_albumentations
import torch

inject_custom_albumentations("grayscale")
model = YOLO("YOLO/base_models/yolov8s.pt")

print("Starting a 1 epoch test run to check if workers use the patch...")
model.train(data="dataset/uavod10.yaml", epochs=1, imgsz=160, batch=2, workers=2, project="test_run", name="test_gray", device="cpu")
