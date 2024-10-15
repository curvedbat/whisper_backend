from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import whisper
import os

app = FastAPI()

# Load Whisper model (you can use 'small', 'base', or 'large' versions)
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe_audio(audioFile: UploadFile = File(...)):
    # Save the uploaded audio file to a temporary location
    audio_path = f"temp/{audioFile.filename}"
    with open(audio_path, "wb") as buffer:
        buffer.write(await audioFile.read())

    # Transcribe the audio using Whisper
    result = model.transcribe(audio_path)

    # Optionally, delete the temporary file after processing
    os.remove(audio_path)

    # Return the transcription
    return JSONResponse({"transcription": result["text"]})
