from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings


class MemoryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_path: Path = settings.data_dir / "feedback_memories.jsonl"
        self.client = self._build_mem0_client()

    def _build_mem0_client(self) -> Any | None:
        if not self.settings.mem0_api_key:
            return None
        try:
            from mem0 import MemoryClient

            kwargs: dict[str, str] = {"api_key": self.settings.mem0_api_key}
            if self.settings.mem0_org_id:
                kwargs["org_id"] = self.settings.mem0_org_id
            if self.settings.mem0_project_id:
                kwargs["project_id"] = self.settings.mem0_project_id
            return MemoryClient(**kwargs)
        except Exception:
            return None

    def remember_feedback(
        self,
        *,
        user_id: str,
        helpful: bool,
        feedback_text: str,
        answer_snapshot: str | None,
    ) -> bool:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        local_record = {
            "user_id": user_id,
            "helpful": helpful,
            "feedback_text": feedback_text,
            "answer_snapshot": answer_snapshot or "",
        }
        with self.local_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(local_record, ensure_ascii=True) + "\n")

        if not self.client:
            return False

        try:
            sentiment = "positive" if helpful else "negative"
            messages = [
                {
                    "role": "user",
                    "content": f"Citizen feedback ({sentiment}): {feedback_text or 'No additional comment provided.'}",
                },
                {
                    "role": "assistant",
                    "content": f"Assistant answer snapshot: {answer_snapshot or 'No snapshot stored.'}",
                },
            ]
            self.client.add(messages, user_id=user_id, version="v2")
            return True
        except Exception:
            return False

    def search_relevant_feedback(self, user_id: str, query: str) -> list[str]:
        memories: list[str] = []
        if self.client:
            try:
                results = self.client.search(query, filters={"user_id": user_id}, version="v2")
                for item in results[:3]:
                    memory_text = item.get("memory") or item.get("text") or ""
                    if memory_text:
                        memories.append(memory_text)
            except Exception:
                pass

        memories.extend(self._search_local(user_id, query))
        deduped: list[str] = []
        seen: set[str] = set()
        for item in memories:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped[:4]

    def _search_local(self, user_id: str, query: str) -> list[str]:
        if not self.local_path.exists():
            return []
        query_tokens = set(self._tokenize(query))
        scored: list[tuple[int, str]] = []
        with self.local_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("user_id") != user_id:
                    continue
                text = f"{entry.get('feedback_text', '')} {entry.get('answer_snapshot', '')}".strip()
                if not text:
                    continue
                overlap = len(query_tokens.intersection(self._tokenize(text)))
                if overlap:
                    scored.append((overlap, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:3]]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())

