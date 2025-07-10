from ultralytics import YOLO
import cv2
import os

model = YOLO('yolo_v11_model/best.pt')
results = model.predict(source="macalister_trim.mp4", stream=True, save=False)

# Optional: Save cropped objects to disk or return them as tensors
output_dir = "cropped_frames"
os.makedirs(output_dir, exist_ok=True)

for frame_idx, r in enumerate(results):
    frame = r.orig_img
    boxes = r.boxes
    names = r.names

    for i, box in enumerate(boxes.xyxy):
        cls_id = int(boxes.cls[i].item())
        label = names[cls_id]
        
        if label in ["kicker", "goalie","goalframe","ball"]:  # filter only important objects
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped = frame[y1:y2, x1:x2]

            # Optional: save cropped image for inspection or dataset
            filename = f"{output_dir}/frame{frame_idx}_{label}_{i}.jpg"
            cv2.imwrite(filename, cropped)