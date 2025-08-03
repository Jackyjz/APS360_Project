from ultralytics import YOLO
import os
import glob
from pathlib import Path
import shutil

# === Config Paths ===
video_dir = 'C:/Users/jacky/Downloads/cxy/penalty kicks edited 208/0xx'

base_dir = os.path.join(video_dir, "processed_data")  # You can name it anything you like

model_path = "C:/Users/jacky/aps360/commitment/yolo_v11_model/best.pt"

img_root = os.path.join(base_dir, 'img')   # Per-class image folders
txt_dir = os.path.join(base_dir, 'txt')    # Flattened txt labels
os.makedirs(img_root, exist_ok=True)
os.makedirs(txt_dir, exist_ok=True)

# === Step 1: Load YOLO model ===
model = YOLO(model_path)

# === Step 2: Find all videos ===
video_paths = glob.glob(os.path.join(video_dir, '*.mp4'))

# === Step 3: Predict on each video ===
for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"🎥 Processing {video_name}...")

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
                keep_imgs = all_imgs

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

print("✅ YOLO done. Now merging into img/{class}/ and txt/ ...")

# === Step 6: Merge all crops and labels ===
predict_folders = glob.glob(os.path.join("runs/detect", "*"))
merged_count = 0

for predict_path in predict_folders:
    crop_dir = os.path.join(predict_path, "crops")
    label_dir = os.path.join(predict_path, "labels")

    if not os.path.isdir(crop_dir):
        continue

    for class_name in os.listdir(crop_dir):
        class_crop_path = os.path.join(crop_dir, class_name)
        if not os.path.isdir(class_crop_path):
            continue

        # Destination class folder: commitment/img/{class}/
        class_output_dir = os.path.join(img_root, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        for file in os.listdir(class_crop_path):
            src_file = os.path.join(class_crop_path, file)
            base_name = file.split('.')[0]
            video_folder = Path(predict_path).name
            safe_prefix = video_folder.replace(" ", "_").replace("(", "").replace(")", "")
            output_name = f"{safe_prefix}_{base_name}"

            try:
                # Copy image to img/{class}/
                dest_img = os.path.join(class_output_dir, output_name + ".jpg")
                shutil.copy2(src_file, dest_img)

                # Copy label to flat txt/ folder
                label_file = os.path.join(label_dir, base_name + ".txt")
                dest_label = os.path.join(txt_dir, output_name + ".txt")
                if os.path.exists(label_file):
                    shutil.copy2(label_file, dest_label)

                merged_count += 1

            except Exception as e:
                print(f"❌ Error copying from {src_file}: {e}")

print(f"✅ Done. {merged_count} crops merged into img/{{class}}/ and txt/")