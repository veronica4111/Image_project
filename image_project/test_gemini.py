from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client()

print("Available models:")
for model in client.models.list():
    print(f"  - {model.name}")
