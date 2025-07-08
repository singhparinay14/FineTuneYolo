import os
from ultralytics import YOLO
from PIL import Image

# === CONFIGURATION ===
IMAGE_DIR = "car_dataset/images"
LABEL_DIR = "car_dataset/labels"
TARGET_CLASS = "car"  # YOLO's default COCO car class
CONFIDENCE_THRESHOLD = 0.3

# === INIT MODEL ===
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt too

def process_directory(split, class_folder):
    image_folder = os.path.join(IMAGE_DIR, split, class_folder)
    label_folder = os.path.join(LABEL_DIR, split, class_folder)
    os.makedirs(label_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"📂 Processing {len(image_files)} images in {image_folder}")

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        results = model.predict(img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + ".txt")
        with open(label_path, "w") as f:
            for box in results.boxes:
                class_id = int(box.cls[0].item())
                name = model.names[class_id]
                if name != TARGET_CLASS:
                    continue

                x_center, y_center, width, height = box.xywhn[0]
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")  # 0 = your class ID
        print(f"✅ Annotated {img_file}")

def auto_annotate():
    for split in ["train", "val"]:
        for class_folder in os.listdir(os.path.join(IMAGE_DIR, split)):
            process_directory(split, class_folder)

if __name__ == "__main__":
    auto_annotate()
