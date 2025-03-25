from fastapi import FastAPI, File, UploadFile
import os
import torch
from transformers import AutoTokenizer
import whisper
from models.text_moderation import classify_text

app = FastAPI()

# Load Whisper model for speech-to-text
whisper_model = whisper.load_model("base")

# Directory to store uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/moderate-text/")
async def moderate_text(text: str):
    return classify_text(text)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    
    # Transcribe the audio file
    transcription = whisper_model.transcribe(file_location)
    
    # Moderate the transcribed text
    moderation_result = await moderate_text(transcription["text"])
    
    return {"filename": file.filename, "transcription": transcription["text"], "moderation_result": moderation_result}
