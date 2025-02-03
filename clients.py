import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDER = "GROQ" # SNOVA, GROQ


if PROVIDER == "SNOVA":
    llm = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=os.getenv("SNOVA_API_KEY"))
    MODEL = "DeepSeek-R1-Distill-Llama-70B"
    # MODEL = "Meta-Llama-3.3-70B-Instruct"


if PROVIDER == "GROQ":
    llm = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv('GROQ_API_KEY'),
    )
    MODEL = 'deepseek-r1-distill-llama-70b' 
    # MODEL = 'llama3-70b-8192'