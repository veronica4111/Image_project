from google import genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
client = genai.Client()
img = Image.new('RGB', (64, 64), 'red')
resp = client.models.generate_content(model='gemini-2.5-flash', contents=['Test image prompt.', img])
print(resp.text[:200])
