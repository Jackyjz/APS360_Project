from ultralytics import YOLO
import os
import glob
from pathlib import Path
import shutil

# === Step 1: Load model ===
model = YOLO('commitment/yolo_v11_model/best.pt')

# === Step 2: Get all video files ===
video_dir = 'C:/Users/jacky/Downloads/cxy/penalty kicks edited 208/0xx'
video_paths = glob.glob(os.path.join(video_dir, '*.mp4'))

# === Step 3: Predict on each video ===
for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"🎥 Processing {video_name}...")

    # Run YOLOv11 on video
    results = model.predict(
        source=video_path,
        conf=0.25,
        save=True,
        save_txt=True,
        save_crop=True,
        stream=False,
        name=video_name
    )

    # === Step 4: Trim crops to last 16 (or 8 fallback) per class ===
    predict_dir = os.path.join("runs/detect", video_name)
    crop_dir = os.path.join(predict_dir, "crops")

    if os.path.exists(crop_dir):
        for class_name in os.listdir(crop_dir):
            class_path = os.path.join(crop_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            all_imgs = sorted(
                os.listdir(class_path),
                key=lambda x: os.path.getmtime(os.path.join(class_path, x))
            )

            total = len(all_imgs)
            if total >= 16:
                keep_imgs = all_imgs[-16:]
            elif total >= 8:
                keep_imgs = all_imgs[-8:]
            else:
                keep_imgs = all_imgs  # keep all

            for img in all_imgs:
                if img not in keep_imgs:
                    os.remove(os.path.join(class_path, img))

            print(f"🧹 {class_name}: {len(keep_imgs)} kept out of {total}")

    # === Step 5: Trim labels to last 16 total ===
    label_dir = os.path.join(predict_dir, "labels")
    if os.path.exists(label_dir):
        all_labels = sorted(
            os.listdir(label_dir),
            key=lambda x: os.path.getmtime(os.path.join(label_dir, x))
        )
        keep_labels = all_labels[-16:]
        for label in all_labels:
            if label not in keep_labels:
                os.remove(os.path.join(label_dir, label))
        print(f"🧹 Labels trimmed to last 16 for {video_name}.")

print("✅ All videos processed. Up to 16 crops per class and 16 labels kept per video.")

# === Step 6: Merge all crops by class ===
predict_root = "runs/detect"
merged_root = "all_crops"
os.makedirs(merged_root, exist_ok=True)

predict_folders = glob.glob(os.path.join(predict_root, "*"))
merged_count = 0

for predict_path in predict_folders:
    crop_dir = os.path.join(predict_path, "crops")
    if not os.path.isdir(crop_dir):
        print(f"⚠️ No crops found in: {predict_path}")
        continue

    for class_name in os.listdir(crop_dir):
        class_crop_path = os.path.join(crop_dir, class_name)
        if not os.path.isdir(class_crop_path):
            continue

        dest_class_dir = os.path.join(merged_root, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)

        for file in os.listdir(class_crop_path):
            src_file = os.path.join(class_crop_path, file)
            video_folder = Path(predict_path).name
            video_safe = video_folder.replace(" ", "_").replace("(", "").replace(")", "")
            dest_filename = f"{video_safe}_{file}"
            dest_path = os.path.join(dest_class_dir, dest_filename)

            try:
                shutil.copy2(src_file, dest_path)
                merged_count += 1 
            except Exception as e:
                print(f"❌ Error copying {src_file} → {dest_path}: {e}")

print(f"✅ Merging complete. {merged_count} crops copied to 'all_crops/'")
