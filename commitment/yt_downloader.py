import yt_dlp

url = input("Enter YouTube URL: ")
output_dir = "C:/Users/jacky/aps360/penalty_videos"

# ydl_opts = {
#     'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
#     'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
#     'merge_output_format': 'mp4',  # Ensures final file is .mp4
# }
ydl_opts = {
    'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best',
    'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
    'merge_output_format': 'mp4',  # Ensures final file is saved as .mp4
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
