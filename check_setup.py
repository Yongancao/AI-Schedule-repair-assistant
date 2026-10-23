import os
from dotenv import load_dotenv

load_dotenv()
print("API key found:", bool(os.getenv("GEMINI_API_KEY")))
print("Model:", os.getenv("GEMINI_MODEL"))

from ortools.sat.python import cp_model
import streamlit, pandas, matplotlib
print("All packages import correctly")

from google import genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.interactions.create(
    model=os.getenv("GEMINI_MODEL"),
    input="Reply with one short sentence confirming you can respond.",
)
print("Gemini says:", response.output_text)