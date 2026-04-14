from dotenv import load_dotenv
load_dotenv()

import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
import groq
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """Record audio from the microphone and save it as an MP3 file."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


stt_model = "whisper-large-v3"


def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY=None):
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ API key is missing. Set GROQ_API_KEY in your environment or .env.\n"
            "Example in .env: GROQ_API_KEY=gsk_..."
        )
    try:
        client = Groq(api_key=api_key)
    except groq.AuthenticationError as e:
        raise RuntimeError(
            f"Groq authentication failed. Verify the key is active.\n(Underlying error: {e})"
        ) from e
    except groq.GroqError as e:
        raise RuntimeError(f"Failed to initialize Groq client: {e}") from e

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )

    return transcription.text
