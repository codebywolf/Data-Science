import whisper
import json
import os

'''
Running Whisper locally, especially with the large-v2 model, is very resource-intensive. It requires a powerful GPU, significant VRAM, and strong processing power. On most personal devices, this can be extremely slow or even infeasible.

To solve this, I ran my code on Google Colab, which provides free access to cloud GPUs. This allowed me to efficiently perform the heavy transcription and translation tasks without overloading my own machine.

Alternatively, the same task can be done using the OpenAI Whisper API, which runs on OpenAI’s optimized servers. The API is much faster, requires no setup on your local machine, and removes the need for specialized hardware.

You can use my Google Colab notebook "unused_code/Google_colab_speech_to_text.ipynb"
'''

# Load the Whisper model (large-v2 is accurate but heavy on resources)
model = whisper.load_model("large-v2")

# Get a list of all audio files inside the "audios" directory
audios = os.listdir("audios")

# Iterate over each audio file in the directory
for audio in audios: 
    # Process only files that follow the "<number>_<title>.mp3" naming format
    if("_" in audio):
        # Extract numeric ID (tutorial/episode number) from filename
        number = audio.split("_")[0]
        
        # Extract title (remove ".mp3" extension by slicing off last 4 chars)
        title = audio.split("_")[1][:-4]
        
        # Debug/trace: print extracted number and title
        print(number, title)
        
        # Run Whisper transcription & translation
        # - Input: audio file
        # - language="hi" → tells model audio is in Hindi
        # - task="translate" → translates Hindi speech into English text
        # - word_timestamps=False → only segment-level timestamps
        result = model.transcribe(
            audio=f"audios/{audio}", 
            # audio="trash/sample.mp3",   # [Optional] 10sec mp3 file for testing/debugging
            language="hi",
            task="translate",
            word_timestamps=False
        )
        
        # Collect structured transcript chunks with metadata
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],   # segment start time
                "end": segment["end"],       # segment end time
                "text": segment["text"]      # transcribed text
            })
        
        # Combine transcript chunks + full text into a single JSON structure
        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        # Save the transcript as a JSON file inside "jsons" directory
        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f)
