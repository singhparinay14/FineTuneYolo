import os
from ultralytics import YOLO

# === CONFIGURATION ===
IMAGE_DIR = "car_dataset/images"
LABEL_DIR = "car_dataset/labels"
CONFIDENCE_THRESHOLD = 0.3

# Class mapping based on your YAML
CLASS_MAP = {
    "Ford_Mustang_GT_Convertible_2020": 0,
    "Audi_R8_2014": 1,
    "Audi_RS6_Avant_2020": 2,
    "BMW_X5_2015": 3,
    "Ferrari_F8_Tributo_2020": 4,
    "Ferrari_F40": 5,
    "Lamborghini_Gallardo_2010": 6,
    "Mercedes_AMG_GT_2015": 7,
    "Porsche_911_2020": 8,
    "Tesla_Cybertruck_2023": 9,
}

# Load model
model = YOLO("yolov8n.pt")

def process_directory(split, class_folder):
    image_folder = os.path.join(IMAGE_DIR, split, class_folder)
    label_folder = os.path.join(LABEL_DIR, split, class_folder)
    os.makedirs(label_folder, exist_ok=True)

    class_id = CLASS_MAP[class_folder]

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"📂 Processing {len(image_files)} images in {image_folder}")

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        results = model.predict(img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + ".txt")
        with open(label_path, "w") as f:
            for box in results.boxes:
                pred_class_id = int(box.cls[0].item())
                name = model.names[pred_class_id]
                if name != "car":  # Only annotate cars
                    continue

                x_center, y_center, width, height = box.xywhn[0]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"✅ Annotated {img_file}")

def auto_annotate():
    for split in ["train", "val"]:
        for class_folder in os.listdir(os.path.join(IMAGE_DIR, split)):
            if class_folder.startswith("."):
                continue
            process_directory(split, class_folder)

if __name__ == "__main__":
    auto_annotate()
