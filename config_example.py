import openai
import os

config = {
    "openai_api_key": "<KEY>",
}

def load():
    openai.api_key = config["openai_api_key"]
    os.environ["OPENAI_API_KEY"] = openai.api_key 