from __future__ import annotations

import re
import time
from collections import defaultdict, deque


PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(system|developer)\s+prompt", re.IGNORECASE),
    re.compile(r"show\s+me\s+the\s+secret", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|guardrails)", re.IGNORECASE),
]


class InMemoryRateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = limit_per_minute
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self.requests[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit_per_minute:
            return False
        window.append(now)
        return True


def sanitize_user_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:2000]


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in PROMPT_INJECTION_PATTERNS)


def mask_identifier(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"

