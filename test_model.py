from ultralytics import YOLO
from PIL import Image

# Load your fine-tuned model
model = YOLO("/Users/jolla/Documents/FineTuneYolo/runs/yolov8s_finetuned_50e/weights/best.pt")

# Path to your test image
source = "test_images/"  # Replace with actual image path

# Run prediction
results = model.predict(source, save=True, show=True, conf=0.4)

# Show annotated results (or you can save them from the returned results)
for result in results:
    result.show()  # This opens a window with predictions
