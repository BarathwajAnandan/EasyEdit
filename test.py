import openai
from dotenv import load_dotenv
import os

load_dotenv()


# model = "deepseek-r1-distill-llama-70b"
# llm_client = openai.OpenAI(
#   base_url="https://api.groq.com/openai/v1",
#   api_key=os.getenv('GROQ_API_KEY'),
# )

model = "Meta-Llama-3.3-70B-Instruct"
llm_client = openai.OpenAI(base_url="https://api.sambanova.ai/v1", api_key=os.getenv("SNOVA_API_KEY"))


# https://github.com/openai/swarm/blob/main/examples/basic/bare_minimum.py
from swarm import Swarm, Agent

swarm_client = Swarm(client=llm_client)

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
    model=model,
    # tool_choice="auto"
)

messages = [{"role": "user", "content": "Hi!"}]
response = swarm_client.run(agent=agent, messages=messages)

print(response.choices[0].message.content)