import os
import requests
from ddgs import DDGS
from PIL import Image
from io import BytesIO

# ==== CONFIGURATION ====
CAR_CLASSES = [
    "Ford Mustang GT Convertible 2020",
    "Audi R8 2014",
    "Audi RS6 Avant 2020",
    "BMW X5 2015",
    "Ferrari F8 Tributo 2020",
    "Ferrari F40",
    "Lamborghini Gallardo 2010",
    "Mercedes AMG GT 2015",
    "Porsche 911 2020",
    "Tesla Cybertruck 2023",
]

IMAGES_PER_CLASS = 50
SAVE_ROOT = "car_dataset"
TRAIN_SPLIT = 0.8  # 80% train, 20% val

# ==== SCRIPT START ====
def create_dirs(class_name):
    train_path = os.path.join(SAVE_ROOT, "images", "train", class_name.replace(" ", "_"))
    val_path = os.path.join(SAVE_ROOT, "images", "val", class_name.replace(" ", "_"))
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    return train_path, val_path

def download_images(class_name, limit=50):
    with DDGS() as ddgs:
        results = ddgs.images(class_name + " car", max_results=limit)
        return [r["image"] for r in results]

def save_image_from_url(url, path):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(path)
        return True
    except Exception:
        return False

def main():
    for car_class in CAR_CLASSES:
        print(f"\n🔍 Downloading images for {car_class}...")
        urls = download_images(car_class, limit=IMAGES_PER_CLASS)
        n_train = int(len(urls) * TRAIN_SPLIT)
        train_path, val_path = create_dirs(car_class)

        count = 0
        for i, url in enumerate(urls):
            name = car_class.replace(" ", "_").lower()
            filename = f"{name}_{i+1:03d}.jpg"
            folder = train_path if i < n_train else val_path
            full_path = os.path.join(folder, filename)
            if save_image_from_url(url, full_path):
                count += 1
                print(f"✅ {filename}")
            else:
                print(f"⚠️ Skipped invalid image: {url}")

        print(f"✅ {count} images saved for {car_class}")

if __name__ == "__main__":
    main()
