"""Chat"""

import logging
import os

import weave
from langchain_core.messages.tool import ToolMessage
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


def chat_with_context(messages: list[dict[str, str]]) -> dict[str, object]:
    """Chat and also return tool-call contexts used as grounding."""
    result = chat_messages(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    )
    msgs = result.get("messages", [])  # type: ignore[reportAny]
    answer = ""
    tool_items: list[dict[str, str]] = []
    try:
        if msgs:
            answer = msgs[-1].content  # type: ignore[reportAttributeAccessIssue]
        for m in msgs:
            if isinstance(m, ToolMessage):
                # name may be None depending on backend; normalize to empty string
                name = getattr(m, "name", "") or ""
                # ToolMessage.content can be str | list | dict; stringify safely
                content_raw = m.content
                if isinstance(content_raw, str):
                    content_str = content_raw
                else:
                    content_str = str(content_raw)
                tool_items.append({"tool": name, "content": content_str})
    except Exception:
        logger.exception("Failed to extract tool messages from chat result")

    return {
        "generated_text": str(answer).strip(),
        "context": {"tool_messages": tool_items},
    }
