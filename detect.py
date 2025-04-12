from ultralytics import YOLO

# Load the model using the correct path
model = YOLO("D:/Vehicle Detection/runs/detect/train2/weights/best.pt")

# Run inference on an image or video
results = model("D:/Vehicle Detection/New folder/Test Video.mp4", save=True)

