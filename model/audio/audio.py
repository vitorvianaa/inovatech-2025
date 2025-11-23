# gerar audio
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import pygame
import time
import os

load_dotenv()
api = os.getenv('API_KEY')
client = OpenAI(api_key=api)
speech_file_path = Path(__file__).parent / "speech.mp3"


def generate_audio(name: str):
    texto = f'Oi, meu nome é {name}!'
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=texto,
        instructions="Speak in a cheerful and positive tone.",
    ) as response:
        response.stream_to_file(speech_file_path)

    return speech_file_path

def play_audio(file_path: str):
    pygame.init()
    pygame.mixer.init()

    try:
        pygame.mixer.music.load(file_path)
    except pygame.error:
        print("Não foi possível carregar o arquivo de áudio.")
        exit()

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(1)

    pygame.quit()