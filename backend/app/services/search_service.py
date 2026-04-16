from __future__ import annotations

import json
import logging
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from backend.app.core.config import Settings
from backend.app.services.guardrails import filter_search_results

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        api_wrapper = TavilySearchAPIWrapper(tavily_api_key=settings.tavily_api_key)
        self.tool = TavilySearchResults(
            api_wrapper=api_wrapper,
            max_results=2,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            include_domains=settings.trusted_search_domains,
        )
        self.fallback_tool = TavilySearchResults(
            api_wrapper=api_wrapper,
            max_results=2,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
        )

    def search(self, queries: list[str]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for query in queries[:1]:
            result = self.tool.invoke({"query": query})
            parsed = self._normalize(result)
            logger.info("tavily_query=%s parsed_results=%s", query, len(parsed))
            filtered = filter_search_results(parsed, self.settings)
            for item in filtered:
                url = item.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                merged.append(item)

        # Fallback: if strict-domain search returns nothing, run a broader search
        # and apply trusted-source filtering afterward.
        if not merged:
            for query in queries[:1]:
                result = self.fallback_tool.invoke({"query": query})
                parsed = self._normalize(result)
                logger.info("tavily_fallback_query=%s parsed_results=%s", query, len(parsed))
                filtered = filter_search_results(parsed, self.settings)
                for item in filtered:
                    url = item.get("url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    merged.append(item)
        return merged[:4]

    def _normalize(self, raw_result: Any) -> list[dict[str, Any]]:
        if isinstance(raw_result, list):
            return [item for item in raw_result if isinstance(item, dict)]
        if isinstance(raw_result, dict):
            # Tavily wrappers can return {"results":[...]} or a single result dict.
            nested = raw_result.get("results")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
            if raw_result.get("url"):
                return [raw_result]
            return []
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
                if isinstance(parsed, dict):
                    nested = parsed.get("results")
                    if isinstance(nested, list):
                        return [item for item in nested if isinstance(item, dict)]
                    if parsed.get("url"):
                        return [parsed]
            except json.JSONDecodeError:
                return []
        return []
