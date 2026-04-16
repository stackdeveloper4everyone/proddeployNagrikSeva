from __future__ import annotations

from urllib.parse import urlparse

from backend.app.core.config import Settings


def is_trusted_source(url: str, settings: Settings) -> bool:
    if not url:
        return False
    hostname = (urlparse(url).hostname or "").lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in settings.trusted_search_domains)


def filter_search_results(results: list[dict], settings: Settings) -> list[dict]:
    trusted = [item for item in results if is_trusted_source(item.get("url", ""), settings)]
    return trusted or results[:3]

