"""
プロンプト改善機能

フィードバックログを基にチャットボットの回答プロンプトを改善する機能を提供します。
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PromptImprovementResult(BaseModel):
    """プロンプト改善結果のスキーマ"""

    improved_prompt: str
    change_reason: str
    feedback_analysis: dict[str, Any]


def load_feedback_logs(csv_path: str | Path) -> list[dict[str, Any]]:
    """フィードバックログをCSVから読み込む"""
    logs = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        logs.extend(
            {
                "id": row["id"],
                "question": row["Question"],
                "answer": row["Answer"],
                "feedback": row["Feedback"],
                "reason": row["Reason"],
            }
            for row in reader
        )
    return logs


def analyze_feedback_patterns_with_llm(
    feedback_logs: list[dict[str, Any]],
) -> dict[str, Any]:
    """LLMを使用してフィードバックログから問題パターンを分析"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    negative_feedbacks = [log for log in feedback_logs if log["feedback"] == "ng"]
    positive_feedbacks = [log for log in feedback_logs if log["feedback"] == "ok"]

    if not negative_feedbacks:
        return {
            "total_logs": len(feedback_logs),
            "negative_count": len(negative_feedbacks),
            "positive_count": len(positive_feedbacks),
            "problem_patterns": {},
            "negative_feedbacks": negative_feedbacks,
            "positive_feedbacks": positive_feedbacks,
        }

    # フィードバックログを分析用のテキストに変換
    feedback_text = ""

    # ネガティブフィードバックを追加
    for i, log in enumerate(negative_feedbacks, 1):
        feedback_text += f"""
ネガティブフィードバック {i}:
質問: {log["question"]}
回答: {log["answer"]}
フィードバック理由: {log["reason"]}
"""

    # ポジティブフィードバックも追加
    for i, log in enumerate(positive_feedbacks, 1):
        feedback_text += f"""
ポジティブフィードバック {i}:
質問: {log["question"]}
回答: {log["answer"]}
フィードバック理由: {log["reason"]}
"""

    system_prompt = """
あなたはカスタマーサポートの品質分析専門家です。
フィードバックログを分析して、チャットボットの回答で発生している問題パターンを特定してください。

分析結果は以下のJSON形式で返してください：
{
    "problem_patterns": {
        "pattern_1": {
            "name": "パターン名",
            "description": "パターンの説明",
            "count": 該当するフィードバック数,
            "examples": ["該当するフィードバックの例1", "例2"]
        },
        "pattern_2": {
            ...
        }
    },
    "summary": "全体的な問題の要約",
    "priority_issues": ["優先的に改善すべき問題1", "問題2"]
}
""".strip()

    user_prompt = f"""
以下のフィードバックログを分析して、チャットボットの回答で発生している問題パターンを特定してください。
ネガティブフィードバックから問題パターンを特定し、ポジティブフィードバックからは良い例を参考にしてください。
問題パターンは具体的で改善可能なものにしてください。
{feedback_text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content or "{}"
    content = content.replace("```json\n", "").replace("\n```", "")

    try:
        analysis_result = json.loads(content)
        return {
            "total_logs": len(feedback_logs),
            "negative_count": len(negative_feedbacks),
            "positive_count": len(positive_feedbacks),
            "problem_patterns": analysis_result.get("problem_patterns", {}),
            "summary": analysis_result.get("summary", ""),
            "priority_issues": analysis_result.get("priority_issues", []),
            "negative_feedbacks": negative_feedbacks,
            "positive_feedbacks": positive_feedbacks,
        }
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse feedback analysis result")
        logger.exception("Response content: %s", content)
        error_msg = f"Feedback analysis parsing failed: {e}"
        raise ValueError(error_msg) from e


def improve_prompt_with_llm(
    current_prompt: str, feedback_analysis: dict[str, Any]
) -> PromptImprovementResult:
    """LLMを使用してプロンプトを改善"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # フィードバック分析結果を文字列に変換
    analysis_text = json.dumps(feedback_analysis, ensure_ascii=False, indent=2)

    system_prompt = """
あなたはカスタマーサポートチャットボットのプロンプト改善専門家です。
フィードバックログの分析結果を基に、現在のプロンプトを改善してください。

改善の際は以下の点を重視してください：
1. ユーザーの不満の原因を解決する
2. サービス品質の向上

回答は必ずJSON形式で返してください。
""".strip()

    user_prompt = f"""
# 現在のプロンプト
{current_prompt}

# フィードバック分析結果
{analysis_text}

上記の分析結果を基に、プロンプトを改善してください。

出力形式：
{{
    "improved_prompt": "改善後のプロンプト（完全なプロンプト）",
    "change_reason": "改善理由の詳細説明（なぜこの変更が必要で、どのような効果が期待されるか）"
}}

改善のポイント：
- 問題パターンに基づいた具体的な改善
- ユーザー体験の向上
""".strip()  # noqa: E501

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content or "{}"

    # JSONの前後のマークダウン記法を除去
    content = content.replace("```json\n", "").replace("\n```", "")

    try:
        result_data = json.loads(content)
        return PromptImprovementResult(
            improved_prompt=result_data["improved_prompt"],
            change_reason=result_data["change_reason"],
            feedback_analysis=feedback_analysis,
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.exception("Failed to parse LLM response")
        logger.exception("Response content: %s", content)
        error_msg = f"LLM response parsing failed: {e}"
        raise ValueError(error_msg) from e


def improve_prompt_from_feedback(
    current_prompt: str, feedback_csv_path: str | Path = "data/feedback-logs.csv"
) -> PromptImprovementResult:
    """
    フィードバックログを基にプロンプトを改善するメイン関数

    Args:
        current_prompt: 現在のプロンプト
        feedback_csv_path: フィードバックログのCSVファイルパス

    Returns:
        PromptImprovementResult: 改善結果

    """
    # フィードバックログを読み込み
    logger.info("Loading feedback logs from %s", feedback_csv_path)
    feedback_logs = load_feedback_logs(feedback_csv_path)
    logger.info("Loaded %d feedback logs", len(feedback_logs))

    # フィードバックパターンを分析
    logger.info("Analyzing feedback patterns with LLM")
    feedback_analysis = analyze_feedback_patterns_with_llm(feedback_logs)
    logger.info(
        "Analysis complete: %d negative, %d positive",
        feedback_analysis["negative_count"],
        feedback_analysis["positive_count"],
    )

    # LLMでプロンプトを改善
    logger.info("Improving prompt with LLM")
    result = improve_prompt_with_llm(current_prompt, feedback_analysis)
    logger.info("Prompt improvement complete")

    return result


if __name__ == "__main__":
    # テスト実行
    from src.prompts import SYSTEM_PROMPT

    result = improve_prompt_from_feedback(SYSTEM_PROMPT)

    logger.info("=== 改善後のプロンプト ===")
    logger.info(result.improved_prompt)
    logger.info("=== 変更理由 ===")
    logger.info(result.change_reason)
    logger.info("=== フィードバック分析 ===")
    output_path = Path("data/output/feedback_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info("Feedback analysis saved to %s", output_path)
