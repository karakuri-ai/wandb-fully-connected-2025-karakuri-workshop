"""
Run Weave Evaluation for operational improvement: detect knowledge gaps.

This script evaluates chatbot responses and flags cases where the bot
explicitly/implicitly indicates lack of knowledge or insufficient context,
to help prioritize FAQ/knowledge base improvements.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import weave
from openai import OpenAI
from weave import Dataset, Evaluation, Model, Table

from src.chat import chat_with_context
from src.config import Config
from src.prompts import prompt_builder_for_improvement_proposal

logger = logging.getLogger(__name__)


def loose_parse_json(content: str) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Loose parse JSON from content.

    Strips markdown fences and attempts json.loads, returning empty dict on error.
    """
    prefix = "```json\n"
    suffix = "\n```"
    content = content.replace(prefix, "").replace(suffix, "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


class ChatBotModel(Model):
    """Model wrapper to obtain chatbot responses."""

    @weave.op()
    def predict(self, question: str) -> dict[str, Any]:
        """Return chatbot response for given question."""
        return chat_with_context([{"role": "user", "content": question}])


def load_questions(
    file_path: str | Path = "data/eval-onboarding.csv",
) -> list[dict[str, Any]]:
    """Load evaluation questions from CSV (id, Question)."""
    questions_path = Path(file_path)
    rows: list[dict[str, Any]] = []
    with questions_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend({"id": r["id"], "question": r["Question"]} for r in reader)
    return rows


class KnowledgeGapScorer(weave.Scorer):
    """
    Detect whether the bot indicates knowledge insufficiency.

    Heuristics judged by LLM-as-a-judge:
    - Bot claims insufficient information / cannot answer from available context
    - Bot repeatedly asks for unspecified details that should be in FAQ
    - Bot suggests contacting support due to missing knowledge, not policy
    Output schema: {"knowledge_gap": {"flag": 0/1, "reason": str}}
    """

    model_id: str = "gpt-4.1"

    def __init__(self) -> None:
        """Initialize OpenAI client."""
        super().__init__()
        self._client = OpenAI()

    @weave.op
    def score(self, output: dict[str, Any], question: str) -> dict:  # type: ignore[reportIncompatibleVariableOverride]
        """Return knowledge gap flag and reason using improvement prompt builder."""
        answer = str(output.get("generated_text", "")).strip()
        # Insert tool contexts if available between user and assistant
        tool_msgs = output["context"]["tool_messages"]
        blocks: list[str] = [
            "role: system\ncontent: (omitted)",
            f"role: user\ncontent: {question}",
        ]
        for t in tool_msgs:
            tool_name = t["tool"]
            content = t["content"]
            header = f"tool({tool_name})"
            blocks.append(f"role: {header}\ncontent: {content}")
        blocks.append(f"role: assistant\ncontent: {answer}")
        conversation_text = "\n\n".join(blocks)
        user_prompt = prompt_builder_for_improvement_proposal(conversation_text)
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": "あなたはサポート運用改善のための評価者です。",
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        parsed = loose_parse_json(content)
        logger.info("KnowledgeGapScorer parsed: %s", parsed)
        # Map improvement JSON to a compact flag/reason
        if isinstance(parsed, dict):
            reason = str(parsed.get("discontent", ""))
            flag = int(parsed.get("lack_of_knowledge", 0))
        else:
            reason = str(parsed)
            flag = 0
        return {"knowledge_gap": {"reason": reason, "flag": flag, "raw": parsed}}


async def main() -> None:
    """Run improvement evaluation for knowledge gaps."""
    config = Config()  # type: ignore[reportCallIssue]
    os.environ["OPENAI_API_KEY"] = config.openai_api_key.get_secret_value()
    os.environ["WANDB_API_KEY"] = config.wandb_api_key.get_secret_value()
    weave.init(config.weave_project_name)

    logger.info("Loading questions...")
    questions = load_questions()
    logger.info("Loaded %d examples", len(questions))

    model = ChatBotModel()
    scorer = KnowledgeGapScorer()

    table = Table([{"question": q["question"], "id": q["id"]} for q in questions])
    dataset = Dataset(name="knowledge_improvement", rows=table)

    evaluation = Evaluation(dataset=dataset, scorers=[scorer])

    logger.info("Starting improvement evaluation...")
    results = await evaluation.evaluate(model)
    logger.info("Improvement evaluation finished: %s", results)


if __name__ == "__main__":
    asyncio.run(main())
