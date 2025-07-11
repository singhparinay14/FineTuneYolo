from collections import Counter
import glob

counts = Counter()
for f in glob.glob("car_dataset/labels/train/**/*.txt", recursive=True):
    with open(f) as file:
        for line in file:
            if line.strip():
                cls_id = int(line.strip().split()[0])
                counts[cls_id] += 1

print("🔢 Class frequency:", dict(counts))
