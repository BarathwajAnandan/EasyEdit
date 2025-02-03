import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
load_dotenv()

PROVIDER = "SNOVA" # SNOVA, GROQ

import os
print("Available secret keys:", list(st.secrets.keys()))
if PROVIDER == "SNOVA":
    api_key = os.getenv("SNOVA_API_KEY") or st.secrets["SNOVA_API_KEY"]
    llm = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=api_key)
    MODEL = "DeepSeek-R1-Distill-Llama-70B"
    # MODEL = "Meta-Llama-3.3-70B-Instruct"

if PROVIDER == "GROQ":
    api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
    llm = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )
    MODEL = 'deepseek-r1-distill-llama-70b' 
    # MODEL = 'llama3-70b-8192'
