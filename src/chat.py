"""Chat"""

import logging
import os

import weave
from langchain_core.runnables.base import Runnable
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

from src.prompts import SYSTEM_PROMPT
from src.tools import semantic_search_faqs

logger = logging.getLogger(__name__)


def setup_weave(wandb_api_key: str, weave_project_name: str) -> None:
    """Weaveのセットアップ"""
    os.environ["WANDB_API_KEY"] = wandb_api_key
    weave.init(weave_project_name)


def setup_agent() -> Runnable:
    """エージェントのセットアップを行う同期関数"""
    model = ChatOpenAI(model="gpt-4.1", api_key=SecretStr(os.environ["OPENAI_API_KEY"]))
    tools = [semantic_search_faqs]
    return create_react_agent(model, tools)


@weave.op()
def chat_messages(messages: list[dict[str, str]]) -> list[dict]:
    """Chat messages"""
    agent = setup_agent()
    return agent.invoke({"messages": messages})


def chat(messages: list[dict[str, str]]) -> str:
    """Chat"""
    response = chat_messages(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    )
    return response["messages"][-1].content  # type: ignore[reportIndexIssue]
