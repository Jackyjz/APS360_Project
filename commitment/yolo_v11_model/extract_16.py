from roboflow import Roboflow
from ultralytics import YOLO
import os
import glob

# Step 1: Run YOLO prediction with crop and label saving
model = YOLO('commitment/yolo_v11_model/best.pt')
results = model.predict(
    source='commitment/macalister_trim.mp4',
    conf=0.25,
    save=True,
    stream=False,
    save_txt=True,
    save_crop=True,
)

# Step 2: Locate the most recent 'predict' output folder
runs_dir = "runs/detect"
predict_dirs = sorted(glob.glob(os.path.join(runs_dir, "predict*")), key=os.path.getmtime)
latest_dir = predict_dirs[-1]  # latest YOLO run directory

# Step 3: Keep last 16 cropped frames per class
crop_dir = os.path.join(latest_dir, "crops")
for class_name in os.listdir(crop_dir):
    class_path = os.path.join(crop_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    all_frames = sorted(
        os.listdir(class_path),
        key=lambda x: os.path.getmtime(os.path.join(class_path, x))
    )
    keep_frames = all_frames[-16:]  # last 16 by modification time

    for frame in all_frames:
        if frame not in keep_frames:
            os.remove(os.path.join(class_path, frame))

# Step 4: Keep last 16 label .txt files (across all classes)
labels_dir = os.path.join(latest_dir, "labels")
all_labels = sorted(
    os.listdir(labels_dir),
    key=lambda x: os.path.getmtime(os.path.join(labels_dir, x))
)
keep_labels = all_labels[-16:]

for label_file in all_labels:
    if label_file not in keep_labels:
        os.remove(os.path.join(labels_dir, label_file))

print("✅ Done: Last 16 crops per class and last 16 label files kept.")
