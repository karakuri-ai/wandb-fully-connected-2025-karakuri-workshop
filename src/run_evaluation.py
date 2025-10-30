"""
Run Weave Evaluation with QAAccuracyScorer.

This script loads the evaluation dataset and runs the evaluation using
the QAAccuracyScorer to assess bot response quality.
"""

import asyncio
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import weave
from openai import OpenAI
from pydantic import BaseModel, Field
from weave import Dataset, Evaluation, Model, Table

from src.chat import chat_with_context
from src.config import Config
from src.prompts import (
    SYSTEM_PROMPT,
    prompt_builder_for_common_sense_verification,
    prompt_builder_for_statement_extraction,
    prompt_builder_for_statement_verification,
)

logger = logging.getLogger(__name__)


class ChatBotModel(Model):
    """Model wrapper for the chat bot responses."""

    @weave.op()
    def predict(self, question: str) -> dict[str, Any]:
        """Return the pre-generated response for the given question ID."""
        return chat_with_context([{"role": "user", "content": question}])


def load_evaluation_dataset(
    file_path: str | Path = "data/eval-onboarding.csv",
) -> list[dict[str, Any]]:
    """Load evaluation dataset from CSV files."""
    # Load questions
    questions_path = Path(file_path)
    questions = []
    with questions_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions.extend(
            {"id": row["id"], "question": row["Question"]} for row in reader
        )

    return questions


class QAAccuracyResult(BaseModel):
    """QAAccuracyResult schema"""

    reason: str = Field(description="ボットの回答の適切性に関する理由")
    answer: int = Field(description="回答の適切性")


def loose_parse_json(content: str) -> dict[str, Any] | list[dict[str, Any]]:
    """Loose parse JSON from content."""
    prefix = "```json\n"
    suffix = "\n```"
    content = content.replace(prefix, "").replace(suffix, "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


class QAAccuracyScorer(weave.Scorer):
    """LLM-as-a-judge scorer for answer appropriateness."""

    model_id: str = "gpt-4.1"
    system_prompt: str = (
        "あなたは厳密な評価者です。指示に従い、日本語で簡潔かつ具体的に"
        "評価してください。"
    )

    def __init__(self) -> None:
        """Initialize OpenAI client for judging."""
        super().__init__()
        self._client = OpenAI()

    @weave.op
    def call_judge(self, question: str, answer: str) -> dict[str, Any]:
        """Call LLM judge and return parsed JSON result."""
        user_prompt_template = """
下記に、ユーザの問い合わせとチャットボットの回答が与えられます。この回答が適切かどうかを判定してください。
回答の適切性に関しては、下記の基準をもとに判定してください。
また、回答は後述するJSON（dict[str, Any]）形式で記述してください。

# 回答の適切性の判断基準
1. 適切に回答を行えた
    a. ユーザの問いかけに対して、意図を汲み取った適切な回答ができている
    b. ユーザの入力に不明瞭さがある場合、その意図を明確化するための適切な聞き返しを行えている
    c. 上記aとbのどちらでもないが、回答としては適切
        - 挨拶や雑談への対応など。
2. 情報が不足しており、質問内容に答えられない。
    a. 参照情報からは回答できない旨を伝えている。
3. 回答を行ったが内容が不適切
    a. ユーザの意図を読み違えており、会話が噛み合っていないなど。
# 入力
ユーザの問い合わせ: {question}
チャットボットの回答: {answer}

# 回答形式（dict[str, Any] 形式）
- key
    - "reason": ボットの回答の適切性に関する理由を記述
    - "answer": 回答の適切性の判断基準の通し番号（1,2,3）から最も適切なものを選択

出力は厳密なJSONのみとし、前後に説明文は一切出力しないでください。
""".strip()  # noqa: E501
        user_prompt = user_prompt_template.format(question=question, answer=answer)

        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        parsed = loose_parse_json(content)
        logger.info("QAAccuracyScorer parsed: %s", parsed)
        return parsed  # type: ignore[reportReturnType]

    @weave.op
    def score(self, output: dict[str, Any], question: str) -> dict:  # type: ignore[reportIncompatibleVariableOverride]
        """Score appropriateness of `output` for the given question (in kwargs)."""
        judged = self.call_judge(question=question, answer=output["generated_text"])
        result = QAAccuracyResult(
            reason=str(judged.get("reason", "")),
            answer=int(judged.get("answer", 2)),
        )
        logger.info("QAAccuracyScorer score result: %s", result.model_dump())
        return {"accuracy": result.model_dump()}


class HallucinationResult(BaseModel):
    """HallucinationResult schema"""

    hallucination: float = Field(description="ハルシネーション割合")
    details: list[dict[str, Any]] = Field(description="詳細結果")


class ContextHallucinationScorer(weave.Scorer):
    """Check that assistant output does not contradict provided context."""

    model_id: str = "gpt-4.1"

    def __init__(self) -> None:
        """Initialize OpenAI client."""
        super().__init__()
        self._client = OpenAI()

    @staticmethod
    def _build_context(
        question: str,
        answer: str,
        tool_messages: list[dict[str, str]],
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        # Insert tool messages as part of context if available
        if tool_messages:
            for t in tool_messages:
                tool_name = t["tool"]
                content = t["content"]
                header = f"tool({tool_name})"
                messages.append({"role": header, "content": content})
        messages.append({"role": "assistant", "content": answer})
        return "\n\n".join(
            [f"role: {m['role']}\ncontent: {m['content']}" for m in messages]
        )

    @weave.op
    def _extract_statements(self, answer: str) -> list[str]:
        prompt = prompt_builder_for_statement_extraction([answer])
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "Extract statements as instructed."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or "[]"
        logger.info("Content: %s", content)
        items = loose_parse_json(content)
        logger.info("ContextHallucinationScorer _extract_statements parsed: %s", items)
        return items  # type: ignore[reportReturnType]

    @weave.op
    def _verify_against_context(
        self, statements: list[str], context: str
    ) -> list[dict[str, Any]]:
        prompt = prompt_builder_for_statement_verification(statements, context)
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Judge statements directly against the provided context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        parsed = loose_parse_json(content)
        logger.info(
            "ContextHallucinationScorer _verify_against_context parsed: %s", parsed
        )
        return parsed  # type: ignore[reportReturnType]

    @weave.op
    def score(self, output: dict[str, Any], question: str) -> dict:  # type: ignore[reportIncompatibleVariableOverride]
        """Produce verdicts by checking output against built context."""
        tool_msgs = output["context"]["tool_messages"]
        context = self._build_context(question, output["generated_text"], tool_msgs)  # type: ignore[reportArgumentType]
        statements = self._extract_statements(output["generated_text"])
        verdicts = self._verify_against_context(statements, context)
        return {
            "context_hallucination": HallucinationResult(
                hallucination=sum(v["validity"] == "0" for v in verdicts)
                / len(verdicts)
                if len(verdicts) > 0
                else 0,
                details=verdicts,
            ).model_dump()
        }


class CommonSenseHallucinationScorer(weave.Scorer):
    """Check that assistant output aligns with common sense rules."""

    model_id: str = "gpt-4.1"

    def __init__(self) -> None:
        """Initialize OpenAI client."""
        super().__init__()
        self._client = OpenAI()

    @weave.op
    def _extract_statements(self, answer: str) -> list[str]:
        prompt = prompt_builder_for_statement_extraction([answer])
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "Extract statements as instructed."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or "[]"
        items = loose_parse_json(content)
        logger.info(
            "CommonSenseHallucinationScorer _extract_statements parsed: %s", items
        )
        return items  # type: ignore[reportReturnType]

    @staticmethod
    def _build_context(
        question: str,
        answer: str,
        tool_messages: list[dict[str, str]],
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        # Insert tool messages as part of context if available
        if tool_messages:
            for t in tool_messages:
                tool_name = t["tool"]
                content = t["content"]
                header = f"tool({tool_name})"
                messages.append({"role": header, "content": content})
        messages.append({"role": "assistant", "content": answer})
        return "\n\n".join(
            [f"role: {m['role']}\ncontent: {m['content']}" for m in messages]
        )

    @weave.op
    def _verify_common_sense(
        self, statements: list[str], context: str
    ) -> list[dict[str, Any]]:
        prompt = prompt_builder_for_common_sense_verification(statements, context)
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": "Judge statements by common sense criteria.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or "[]"
        parsed = loose_parse_json(content)
        logger.info(
            "CommonSenseHallucinationScorer _verify_common_sense parsed: %s", parsed
        )
        return parsed  # type: ignore[reportReturnType]

    @weave.op
    def score(self, output: dict[str, Any], question: str) -> dict:  # type: ignore[reportIncompatibleVariableOverride]
        """Produce verdicts by checking output against common sense rules."""
        tool_msgs = output["context"]["tool_messages"]
        context = self._build_context(question, output["generated_text"], tool_msgs)  # type: ignore[reportArgumentType]
        statements = self._extract_statements(output["generated_text"])
        verdicts = self._verify_common_sense(statements, context)
        return {
            "common_sense_hallucination": HallucinationResult(
                hallucination=sum(v["validity"] != "0" for v in verdicts)
                / len(verdicts)
                if len(verdicts) > 0
                else 0,
                details=verdicts,
            ).model_dump()
        }


async def main() -> None:
    """Run the evaluation using Weave Evaluation."""
    config = Config()  # type: ignore[reportCallIssue]
    os.environ["OPENAI_API_KEY"] = config.openai_api_key.get_secret_value()
    os.environ["WANDB_API_KEY"] = config.wandb_api_key.get_secret_value()
    weave.init(config.weave_project_name)

    logger.info("Loading evaluation dataset...")
    dataset = load_evaluation_dataset()
    logger.info("Loaded %d evaluation examples", len(dataset))

    qa_accuracy_scorer = QAAccuracyScorer()
    context_scorer = ContextHallucinationScorer()
    commonsense_scorer = CommonSenseHallucinationScorer()
    model = ChatBotModel()

    # Create evaluation
    table = Table(dataset)
    weave_dataset = Dataset(name="chatbot_evaluation", rows=table)
    evaluation = Evaluation(
        dataset=weave_dataset,
        scorers=[qa_accuracy_scorer, context_scorer, commonsense_scorer],
    )

    logger.info("Starting evaluation...")
    try:
        results = await evaluation.evaluate(model)
        logger.info("Evaluation completed successfully")
        logger.info("Results: %s", results)
    except Exception:
        logger.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
