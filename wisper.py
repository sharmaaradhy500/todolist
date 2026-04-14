import whisper
import sounddevice as sd
import numpy as np
import queue
import threading

# Load model
model = whisper.load_model("tiny")

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def process_audio():
    audio_buffer = np.zeros((0,), dtype=np.float32)

    while True:
        data = q.get().flatten()

        # Normalize audio (IMPORTANT)
        data = data.astype(np.float32)

        audio_buffer = np.concatenate((audio_buffer, data))

        # Process every ~2 seconds (IMPORTANT)
        if len(audio_buffer) >= 32000:
            chunk = audio_buffer[:32000]
            audio_buffer = audio_buffer[32000:]

            result = model.transcribe(chunk, fp16=False)

            text = result["text"].strip()
            if text:
                print("You said:", text)

# Start mic
stream = sd.InputStream(
    samplerate=16000,
    channels=1,
    dtype='float32',   # 🔥 VERY IMPORTANT
    callback=audio_callback
)

with stream:
    threading.Thread(target=process_audio, daemon=True).start()
    print("🎤 Speak now (real-time)...")
    while True:
        pass