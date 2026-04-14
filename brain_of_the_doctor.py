from dotenv import load_dotenv
load_dotenv()

import os
import base64
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

model = "meta-llama/llama-4-scout-17b-16e-instruct"


def encode_image(image_path):
    """Encode a local image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image_with_query(query, model, encoded_image):
    """Send an image + text query to the Groq multimodal LLM and return the response."""
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ API key is missing. Set GROQ_API_KEY in your environment or .env.\n"
            "Example in .env: GROQ_API_KEY=gsk_..."
        )
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq client: {e}") from e

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ],
        }
    ]

    chat_completion = client.chat.completions.create(messages=messages, model=model)
    return chat_completion.choices[0].message.content
