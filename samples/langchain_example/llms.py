from __future__ import annotations

import os

from langchain.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv
load_dotenv()


def get_chat_model() -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(
                    model="qwen3.5:2b",
                    model_kwargs={
                    "reasoning_effort": "low"
                }
            )
    except Exception as e:
        print(f"Error: {e}")


    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("CHAT_DEPLOYMENT_NAME")
    api_version = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

    if not api_key or not endpoint or not deployment:
        raise RuntimeError(
            "Missing Azure OpenAI config. Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "and CHAT_DEPLOYMENT_NAME in .env"
        )

    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        temperature=0,
    )
