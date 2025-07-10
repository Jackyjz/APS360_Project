import os
import subprocess

input_folder = "penalty_videos"
frames_folder = "extracted_frames"
fps = 10  # frames per second

os.makedirs(frames_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        frame_output_path = os.path.join(frames_folder, base_name)
        os.makedirs(frame_output_path, exist_ok=True)

        extract_cmd = [
            "ffmpeg", "-i", input_path,
            "-vf", f"fps={fps}",
            os.path.join(frame_output_path, "frame_%04d.jpg")
        ]
        subprocess.run(extract_cmd, check=True)

        print(f"Extracted frames from: {filename}")
