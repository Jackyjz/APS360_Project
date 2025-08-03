import os
import shutil
import glob
from pathlib import Path

# Where your YOLO runs are stored
predict_root = "runs/detect"
# Where you want to merge all crops
merged_root = "all_crops"
os.makedirs(merged_root, exist_ok=True)

# Look through each YOLO predict folder (e.g., bar messi, Euros italy vs spain)
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

        # Create destination folder for this class
        dest_class_dir = os.path.join(merged_root, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)

        for file in os.listdir(class_crop_path):
            src_file = os.path.join(class_crop_path, file)

            # Make a safe filename using the original folder name
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
