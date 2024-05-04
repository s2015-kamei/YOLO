from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load model

results = model.train(
    data='/home/roboken03/YOLO/datasets/data.yaml',  # Dataset
    epochs=500,
    batch=32,
)

