"""Jev-based relevance gate for automatic note updates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

from tinyCode.config.models import NoteRoutingConfig
from tinyCode.network import detect_proxy_route
from tinyCode.notes.categories import NOTE_CATEGORY_DESCRIPTIONS
from tinyCode.providers.base import TokenUsage


MAX_NOTE_ROUTING_STATE_CHARS = 8_000

_QUESTION_TO_CATEGORY = {
    "user_preferences": "用户偏好",
    "corrections": "纠正反馈",
    "project_knowledge": "项目知识",
    "references": "参考资料",
}


@dataclass(frozen=True)
class NoteRoutingDecision:
    """Categories that contain confidently relevant new information."""

    categories: frozenset[str]
    probabilities: dict[str, float] = field(default_factory=dict)
    usage: TokenUsage = field(default_factory=TokenUsage)


class JevNoteRouter:
    """Select note categories with one batched System One request.

    Noul has no separate confidence field. A probability is actionable only
    when it is close enough to either endpoint. Any answer in the uncertainty
    band rejects the whole gate so the caller can conservatively update every
    category with the existing generative path.
    """

    def __init__(self, config: NoteRoutingConfig) -> None:
        self._config = config

    async def route(self, recent_text: str) -> NoteRoutingDecision:
        endpoint = self._config.base_url.rstrip("/")
        if not endpoint.endswith("/v1/systemone"):
            endpoint += "/v1/systemone"
        proxy_route = detect_proxy_route(endpoint)
        timeout = httpx.Timeout(
            self._config.timeout_seconds,
            connect=min(5.0, self._config.timeout_seconds),
        )
        state = {
            "recent_conversation": recent_text[-MAX_NOTE_ROUTING_STATE_CHARS:],
        }
        questions = {
            question_id: {
                "type": "noul",
                "instructions": (
                    "Does `recent_conversation` contain new, durable information "
                    f"worth saving in the TinyCode note category '{category}'? "
                    f"Category definition: {NOTE_CATEGORY_DESCRIPTIONS[category]}. "
                    "Answer yes only for concrete information that will be useful "
                    "in a future conversation; greetings, transient task progress, "
                    "and facts already merely repeated do not qualify."
                ),
            }
            for question_id, category in _QUESTION_TO_CATEGORY.items()
        }
        payload = {
            "state": state,
            "model": self._config.model,
            "questions": questions,
        }
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            timeout=timeout,
            trust_env=proxy_route.trust_env,
        ) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
        if response.status_code < 200 or response.status_code >= 300:
            detail = response.text[:500].strip() or "无错误详情"
            raise RuntimeError(f"Jev HTTP {response.status_code}: {detail}")
        try:
            raw = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("Jev 返回了无效 JSON") from exc
        if not isinstance(raw, dict):
            raise RuntimeError("Jev 响应顶层不是对象")
        answers = raw.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError("Jev 响应缺少 answers")

        probabilities: dict[str, float] = {}
        selected: set[str] = set()
        threshold = self._config.confidence_threshold
        negative_threshold = 1.0 - threshold
        uncertain: list[str] = []
        for question_id, category in _QUESTION_TO_CATEGORY.items():
            answer = answers.get(question_id)
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                raise RuntimeError(f"Jev 响应缺少 {question_id} Noul 答案")
            value = answer.get("noul")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= float(value) <= 1
            ):
                raise RuntimeError(f"Jev 返回了无效 {question_id} 概率")
            probability = float(value)
            probabilities[category] = probability
            if probability >= threshold:
                selected.add(category)
            elif probability > negative_threshold:
                uncertain.append(f"{category}={probability:.3f}")

        if uncertain:
            raise RuntimeError(
                "Jev 笔记分类置信度不足: " + ", ".join(uncertain)
            )
        return NoteRoutingDecision(
            categories=frozenset(selected),
            probabilities=probabilities,
            usage=TokenUsage.from_raw(raw.get("usage")),
        )
