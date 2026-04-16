from __future__ import annotations

import json
import logging
import re
from typing import Iterable

import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.app.core.config import Settings


logger = logging.getLogger(__name__)


class SarvamClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.sarvam_base_url.rstrip("/")
        self.headers = {
            "api-subscription-key": settings.sarvam_api_key,
            "Content-Type": "application/json",
        }

    def _message_to_payload(self, message: BaseMessage) -> dict[str, str]:
        role = "user"
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        return {"role": role, "content": str(message.content)}

    def chat(self, messages: Iterable[BaseMessage], temperature: float = 0.2) -> str:
        payload_messages = [self._message_to_payload(message) for message in messages]
        return self.chat_messages(payload_messages, temperature=temperature)

    def chat_messages(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        payload = {
            "model": self.settings.sarvam_chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.settings.sarvam_chat_max_tokens,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.settings.sarvam_chat_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        return self._extract_text(body)

    def detect_language(self, text: str) -> str:
        response = requests.post(
            f"{self.base_url}/text-lid",
            headers=self.headers,
            json={"input": text[:1000]},
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        return body.get("language_code") or "en-IN"

    def speech_to_text(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        language_code: str = "auto",
    ) -> tuple[str, str]:
        files = {
            "file": (filename or "audio.webm", file_bytes, content_type or "audio/webm"),
        }
        data = {
            "model": self.settings.sarvam_stt_model,
            "mode": "transcribe",
        }
        if language_code and language_code != "auto":
            data["language_code"] = language_code

        response = requests.post(
            f"{self.base_url}/speech-to-text",
            headers={"api-subscription-key": self.settings.sarvam_api_key},
            files=files,
            data=data,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()

        transcript = (
            body.get("transcript")
            or body.get("text")
            or body.get("transcript_text")
            or body.get("recognized_text")
            or ""
        )
        detected_language = (
            body.get("language_code")
            or body.get("detected_language")
            or language_code
            or "en-IN"
        )
        return transcript.strip(), detected_language

    def text_to_speech(
        self,
        *,
        text: str,
        language_code: str,
    ) -> tuple[str, str]:
        payload = {
            "text": text[:6000],
            "target_language_code": language_code,
            "model": self.settings.sarvam_tts_model,
            "speaker": self.settings.sarvam_tts_speaker,
        }
        response = requests.post(
            f"{self.base_url}/text-to-speech",
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()

        audio_base64 = body.get("audio") or body.get("audio_base64") or ""
        if not audio_base64 and isinstance(body.get("audios"), list) and body["audios"]:
            first = body["audios"][0]
            if isinstance(first, str):
                audio_base64 = first
            elif isinstance(first, dict):
                audio_base64 = first.get("audio") or first.get("audio_base64") or ""

        mime = body.get("audio_mime_type") or "audio/wav"
        return audio_base64, mime

    def translate(self, text: str, source_language_code: str, target_language_code: str) -> tuple[str, str]:
        if not text.strip():
            return text, source_language_code

        source_code = self._normalize_language_code(source_language_code)
        target_code = self._normalize_language_code(target_language_code)
        if source_code == target_code:
            return text, source_code

        parts = self._split_for_translation(text, max_chars=1800)
        translated_parts: list[str] = []
        detected_source = source_code

        for part in parts:
            translated_part, detected_source = self._translate_part_with_retries(
                part=part,
                source_language_code=detected_source,
                target_language_code=target_code,
            )
            translated_parts.append(translated_part)

        return "".join(translated_parts), detected_source

    def _extract_text(self, body: dict) -> str:
        choices = body.get("choices") or []
        if not choices:
            raise ValueError(f"Sarvam response did not include choices: {json.dumps(body)[:500]}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            flattened_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    flattened_parts.append(item)
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("content")
                    if isinstance(text_value, str) and text_value.strip():
                        flattened_parts.append(text_value.strip())
            flattened = "\n".join(part for part in flattened_parts if part).strip()
            if flattened:
                return flattened

        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            return refusal.strip()

        raise ValueError(f"Sarvam response did not include usable assistant text: {json.dumps(body)[:500]}")

    def _split_for_translation(self, text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        paragraphs = re.split(r"(\n\s*\n)", text)
        chunks: list[str] = []
        current = ""

        for part in paragraphs:
            if len(current) + len(part) <= max_chars:
                current += part
                continue

            if current:
                chunks.append(current)
                current = ""

            if len(part) <= max_chars:
                current = part
                continue

            sentences = re.split(r"(?<=[.!?])\s+", part)
            sentence_chunk = ""
            for sentence in sentences:
                if len(sentence_chunk) + len(sentence) + 1 <= max_chars:
                    sentence_chunk = f"{sentence_chunk} {sentence}".strip()
                else:
                    if sentence_chunk:
                        chunks.append(sentence_chunk)
                    sentence_chunk = sentence
            if sentence_chunk:
                current = sentence_chunk

        if current:
            chunks.append(current)

        return chunks or [text[:max_chars]]

    def _translate_part_with_retries(
        self,
        *,
        part: str,
        source_language_code: str,
        target_language_code: str,
    ) -> tuple[str, str]:
        source_variants = self._code_variants(source_language_code, include_auto=True)
        target_variants = self._code_variants(target_language_code, include_auto=False)
        attempts: list[tuple[str, str]] = []
        for src in source_variants:
            for tgt in target_variants:
                if src and tgt and src != tgt:
                    attempts.append((src, tgt))

        last_error: Exception | None = None
        for src, tgt in attempts:
            try:
                response = requests.post(
                    f"{self.base_url}/translate",
                    headers=self.headers,
                    json={
                        "input": part,
                        "source_language_code": src,
                        "target_language_code": tgt,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                body = response.json()
                translated_text = self._extract_translated_text(body)
                if isinstance(translated_text, str) and translated_text.strip():
                    # Guard against no-op responses when target differs.
                    if translated_text.strip() == part.strip() and src.lower() != tgt.lower():
                        continue
                    detected_source = body.get("source_language_code", src)
                    return translated_text, detected_source
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            logger.exception("translation_failed source=%s target=%s", source_language_code, target_language_code)
            raise last_error
        return part, source_language_code

    def _extract_translated_text(self, body: dict) -> str:
        direct = body.get("translated_text") or body.get("translation") or body.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        translations = body.get("translations")
        if isinstance(translations, list) and translations:
            first = translations[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict):
                nested = first.get("translated_text") or first.get("text") or first.get("translation")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()

        output = body.get("output")
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict):
                nested = first.get("translated_text") or first.get("text") or first.get("translation")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()

        return ""

    def _normalize_language_code(self, code: str | None) -> str:
        if not code:
            return "en-IN"
        normalized = code.strip()
        if not normalized:
            return "en-IN"
        return normalized

    def _code_variants(self, code: str, *, include_auto: bool) -> list[str]:
        clean = self._normalize_language_code(code)
        short = clean.split("-")[0].lower()
        variants: list[str] = [clean, short]
        if include_auto:
            variants.append("auto")
        deduped: list[str] = []
        seen: set[str] = set()
        for item in variants:
            key = item.lower()
            if key not in seen and item:
                seen.add(key)
                deduped.append(item)
        return deduped
