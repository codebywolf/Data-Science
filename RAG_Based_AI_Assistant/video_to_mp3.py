import os 
import subprocess

# Get a list of all files inside the "videos" directory
files = os.listdir("videos") 

# Iterate over each file found in the "videos" directory
for file in files: 
    # Extract tutorial number from the filename
    # Example: "Tutorial #12 [Part 1].mp4" → tutorial_number = "12"
    tutorial_number = file.split(" [")[0].split(" #")[1]
    
    # Extract base file name before the separator " ｜ "
    # Example: "Tutorial #12 ｜ Intro.mp4" → file_name = "Tutorial #12"
    file_name = file.split(" ｜ ")[0]
    
    # Debug/trace: print extracted tutorial number and file name
    print(tutorial_number, file_name)
    
    # Use ffmpeg (via subprocess) to convert video to mp3 audio
    # Input:  videos/<original_file>
    # Output: audios/<tutorial_number>_<file_name>.mp3
    subprocess.run([
        "ffmpeg", 
        "-i", f"videos/{file}", 
        f"audios/{tutorial_number}_{file_name}.mp3"
    ])
