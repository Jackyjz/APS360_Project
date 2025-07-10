from roboflow import Roboflow
import ultralytics
from ultralytics import YOLO


model = YOLO('commitment/yolo_v11_model/best.pt')
results = model.predict(
    source='commitment/macalister_trim.mp4',   # path to your video
    conf=0.25,
    save=True,
    stream=False,
    save_txt=True,
    save_crop=True,
)