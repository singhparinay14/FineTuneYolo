import os
import requests
import time
from ddgs import DDGS
from PIL import Image
from io import BytesIO
from ddgs.exceptions import RatelimitException

# ==== CONFIGURATION ====
CAR_CLASSES = [
#    "Ford Mustang GT Convertible 2020",
#    "Audi R8 2014",
#    "Audi RS6 Avant 2020",
#    "BMW X5 2015",
#    "Ferrari F8 Tributo 2020",
#    "Ferrari F40",
#    "Lamborghini Gallardo 2010",
#    "Mercedes AMG GT 2015",
#    "Porsche 911 2020",
    "Tesla Cybertruck 2023",
]

ANGLES = ["front view", "rear view", "side view", "top view"]
LIGHTING = ["sunny", "night", "studio", "indoor"]
BACKGROUNDS = ["on road", "in city", "in garage", "in showroom"]

QUERIES_PER_MODEL = 12
IMAGES_PER_QUERY = 10
SAVE_ROOT = "car_dataset"
TRAIN_SPLIT = 0.8

# ==== HELPERS ====
def create_dirs(class_name):
    train_path = os.path.join(SAVE_ROOT, "images", "train", class_name.replace(" ", "_"))
    val_path = os.path.join(SAVE_ROOT, "images", "val", class_name.replace(" ", "_"))
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    return train_path, val_path

def get_next_index(folder, prefix):
    existing = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".jpg")]
    indices = [int(f.split("_")[-1].split(".")[0]) for f in existing if f.split("_")[-1].split(".")[0].isdigit()]
    return max(indices, default=0) + 1

def generate_queries(base):
    queries = set()
    for angle in ANGLES:
        for light in LIGHTING:
            for bg in BACKGROUNDS:
                query = f"{base} {angle} {light} {bg}"
                queries.add(query)
                if len(queries) >= QUERIES_PER_MODEL:
                    return list(queries)
    return list(queries)

def download_images(search_query, limit=10, retries=3, wait_time=120):
    attempt = 0
    while attempt < retries:
        try:
            with DDGS() as ddgs:
                results = ddgs.images(search_query, max_results=limit)
                return [r["image"] for r in results]
        except RatelimitException:
            print(f"⚠️ Rate limited on attempt {attempt+1} for query: '{search_query}'. Waiting {wait_time} sec...")
            time.sleep(wait_time)
            attempt += 1
            wait_time *= 2
    print(f"❌ Failed to download images for '{search_query}' after {retries} retries.")
    return []

def save_image_from_url(url, path):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(path)
        return True
    except Exception:
        return False

# ==== MAIN ====
def main():
    for car_class in CAR_CLASSES:
        print(f"\n🔍 Downloading images for {car_class}...")
        queries = generate_queries(car_class)
        urls = []

        for query in queries:
            print(f"➡️ Query: {query}")
            query_urls = download_images(query, limit=IMAGES_PER_QUERY)
            urls.extend(query_urls)
            print(f"✅ {len(query_urls)} images from query: '{query}'")
            time.sleep(10)

        print(f"📦 Total URLs collected: {len(urls)}")
        n_train = int(len(urls) * TRAIN_SPLIT)
        train_path, val_path = create_dirs(car_class)

        name = car_class.replace(" ", "_").lower()
        next_train_index = get_next_index(train_path, name)
        next_val_index = get_next_index(val_path, name)

        count = 0
        for i, url in enumerate(urls):
            folder = train_path if i < n_train else val_path
            index = next_train_index if i < n_train else next_val_index
            filename = f"{name}_{index:03d}.jpg"
            full_path = os.path.join(folder, filename)

            if save_image_from_url(url, full_path):
                count += 1
                print(f"✅ Saved: {filename}")
                if i < n_train:
                    next_train_index += 1
                else:
                    next_val_index += 1
            else:
                print(f"⚠️ Skipped invalid image: {url}")

        print(f"✅ {count} new images saved for {car_class}")
        print("⏳ Waiting 30 seconds to avoid rate limiting...")
        time.sleep(30)

if __name__ == "__main__":
    main()
