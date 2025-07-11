import os
import shutil

INPUT_LABEL_DIR = "car_dataset/labels"
OUTPUT_LABEL_DIR = "car_dataset/labels_flat"

os.makedirs(os.path.join(OUTPUT_LABEL_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_LABEL_DIR, "val"), exist_ok=True)

for split in ["train", "val"]:
    split_input = os.path.join(INPUT_LABEL_DIR, split)
    split_output = os.path.join(OUTPUT_LABEL_DIR, split)

    for class_folder in os.listdir(split_input):
        class_path = os.path.join(split_input, class_folder)
        if os.path.isdir(class_path):
            for label_file in os.listdir(class_path):
                src = os.path.join(class_path, label_file)
                dst = os.path.join(split_output, label_file)
                shutil.copyfile(src, dst)

print("✅ Labels successfully flattened!")
