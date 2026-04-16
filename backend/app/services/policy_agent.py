from __future__ import annotations

import json
import logging
import re

from backend.app.models.schemas import ChatResponse, ConversationAnalysis, SourceItem
from backend.app.prompts.templates import FINAL_RESPONSE_PROMPT
from backend.app.services.sarvam_client import SarvamClient
from backend.app.services.session_store import SessionState

logger = logging.getLogger(__name__)


class PolicyAgent:
    LANGUAGE_CODE_MAP = {
        "en": "en-IN",
        "en-in": "en-IN",
        "hi": "hi-IN",
        "hi-in": "hi-IN",
        "ta": "ta-IN",
        "ta-in": "ta-IN",
        "te": "te-IN",
        "te-in": "te-IN",
        "kn": "kn-IN",
        "kn-in": "kn-IN",
        "ml": "ml-IN",
        "ml-in": "ml-IN",
        "mr": "mr-IN",
        "mr-in": "mr-IN",
        "gu": "gu-IN",
        "gu-in": "gu-IN",
        "bn": "bn-IN",
        "bn-in": "bn-IN",
        "pa": "pa-IN",
        "pa-in": "pa-IN",
        "od": "od-IN",
        "od-in": "od-IN",
    }
    LANGUAGE_NAME_MAP = {
        "en-IN": "English",
        "hi-IN": "Hindi",
        "ta-IN": "Tamil",
        "te-IN": "Telugu",
        "kn-IN": "Kannada",
        "ml-IN": "Malayalam",
        "mr-IN": "Marathi",
        "gu-IN": "Gujarati",
        "bn-IN": "Bengali",
        "pa-IN": "Punjabi",
        "od-IN": "Odia",
    }
    RESPONSE_LANGUAGE_HINTS = {
        "hi-IN": ["in hindi", "hindi me", "hindi mein", "hindi", "हिंदी"],
        "ta-IN": ["in tamil", "tamil", "தமிழ்"],
        "te-IN": ["in telugu", "telugu", "తెలుగు"],
        "kn-IN": ["in kannada", "kannada", "ಕನ್ನಡ"],
        "ml-IN": ["in malayalam", "malayalam", "മലയാളം"],
        "mr-IN": ["in marathi", "marathi", "मराठी"],
        "gu-IN": ["in gujarati", "gujarati", "ગુજરાતી"],
        "bn-IN": ["in bengali", "bengali", "বাংলা"],
        "pa-IN": ["in punjabi", "punjabi", "ਪੰਜਾਬੀ"],
        "od-IN": ["in odia", "in oriya", "odia", "oriya", "ଓଡ଼ିଆ"],
        "en-IN": ["in english", "english"],
    }

    STATE_ALIASES = {
        "andhra pradesh": "Andhra Pradesh",
        "arunachal pradesh": "Arunachal Pradesh",
        "assam": "Assam",
        "bihar": "Bihar",
        "chhattisgarh": "Chhattisgarh",
        "goa": "Goa",
        "gujarat": "Gujarat",
        "haryana": "Haryana",
        "himachal pradesh": "Himachal Pradesh",
        "jharkhand": "Jharkhand",
        "karnataka": "Karnataka",
        "kerala": "Kerala",
        "madhya pradesh": "Madhya Pradesh",
        "maharashtra": "Maharashtra",
        "manipur": "Manipur",
        "meghalaya": "Meghalaya",
        "mizoram": "Mizoram",
        "nagaland": "Nagaland",
        "odisha": "Odisha",
        "orissa": "Odisha",
        "punjab": "Punjab",
        "rajasthan": "Rajasthan",
        "sikkim": "Sikkim",
        "tamil nadu": "Tamil Nadu",
        "telangana": "Telangana",
        "tripura": "Tripura",
        "uttar pradesh": "Uttar Pradesh",
        "uttarakhand": "Uttarakhand",
        "west bengal": "West Bengal",
        "andaman and nicobar islands": "Andaman and Nicobar Islands",
        "chandigarh": "Chandigarh",
        "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
        "delhi": "Delhi",
        "nct of delhi": "Delhi",
        "jammu and kashmir": "Jammu and Kashmir",
        "ladakh": "Ladakh",
        "lakshadweep": "Lakshadweep",
        "puducherry": "Puducherry",
        "pondicherry": "Puducherry",
    }

    STATE_ALIASES_HI = {
        "आंध्र प्रदेश": "Andhra Pradesh",
        "अरुणाचल प्रदेश": "Arunachal Pradesh",
        "असम": "Assam",
        "बिहार": "Bihar",
        "छत्तीसगढ़": "Chhattisgarh",
        "गोवा": "Goa",
        "गुजरात": "Gujarat",
        "हरियाणा": "Haryana",
        "हिमाचल प्रदेश": "Himachal Pradesh",
        "झारखंड": "Jharkhand",
        "कर्नाटक": "Karnataka",
        "केरल": "Kerala",
        "मध्य प्रदेश": "Madhya Pradesh",
        "महाराष्ट्र": "Maharashtra",
        "मणिपुर": "Manipur",
        "मेघालय": "Meghalaya",
        "मिजोरम": "Mizoram",
        "नागालैंड": "Nagaland",
        "ओडिशा": "Odisha",
        "उड़ीसा": "Odisha",
        "पंजाब": "Punjab",
        "राजस्थान": "Rajasthan",
        "सिक्किम": "Sikkim",
        "तमिलनाडु": "Tamil Nadu",
        "तमिल नाडु": "Tamil Nadu",
        "तेलंगाना": "Telangana",
        "त्रिपुरा": "Tripura",
        "उत्तर प्रदेश": "Uttar Pradesh",
        "उत्तराखंड": "Uttarakhand",
        "पश्चिम बंगाल": "West Bengal",
        "अंडमान और निकोबार द्वीप समूह": "Andaman and Nicobar Islands",
        "चंडीगढ़": "Chandigarh",
        "दादरा और नगर हवेली और दमन और दीव": "Dadra and Nagar Haveli and Daman and Diu",
        "दिल्ली": "Delhi",
        "जम्मू और कश्मीर": "Jammu and Kashmir",
        "लद्दाख": "Ladakh",
        "लक्षद्वीप": "Lakshadweep",
        "पुदुच्चेरी": "Puducherry",
        "पांडिचेरी": "Puducherry",
    }

    PROFILE_KEYWORDS = {
        "student": "student",
        "teacher": "teacher",
        "school teacher": "teacher",
        "government school teacher": "teacher",
        "govt school teacher": "teacher",
        "professor": "teacher",
        "lecturer": "teacher",
        "lawyer": "lawyer",
        "advocate": "lawyer",
        "attorney": "lawyer",
        "farmer": "farmer",
        "woman": "woman",
        "women": "woman",
        "widow": "widow",
        "senior citizen": "senior citizen",
        "elderly": "senior citizen",
        "unemployed": "unemployed",
        "worker": "worker",
        "labourer": "labourer",
        "laborer": "labourer",
        "entrepreneur": "entrepreneur",
        "business owner": "business owner",
        "self employed": "self employed",
        "self-employed": "self employed",
        "disabled": "person with disability",
        "disability": "person with disability",
        "pwd": "person with disability",
        "pregnant": "pregnant woman",
        "mother": "mother",
        "girl": "girl student",
        "sc": "scheduled caste",
        "st": "scheduled tribe",
        "obc": "obc",
    }

    def __init__(self, sarvam: SarvamClient, search_service, vector_store, memory_service) -> None:
        self.sarvam = sarvam
        self.search_service = search_service
        self.vector_store = vector_store
        self.memory_service = memory_service

    def handle_message(self, session: SessionState, user_message: str, language_code: str) -> ChatResponse:
        response_language = self._resolve_response_language(session, user_message, language_code)
        logger.info("response_language_selected=%s requested_language=%s", response_language, language_code)
        detected_language = response_language
        normalized_message = user_message
        if response_language != "en-IN":
            try:
                normalized_message, detected_source_language = self.sarvam.translate(
                    user_message,
                    response_language,
                    "en-IN",
                )
                detected_language = self._normalize_language_code(detected_source_language or response_language)
            except Exception:
                normalized_message = user_message

        analysis = self._analyze(session, normalized_message, user_message)
        session.intent = analysis.intent
        session.detected_language = detected_language
        session.response_language = response_language
        session.collected_details.update({k: v for k, v in analysis.collected_details.items() if v})
        session.history.append({"role": "user", "content": user_message})

        if not analysis.ready_for_search:
            answer = analysis.clarifying_question or "Please share a few more details so I can help accurately."
            localized_answer = self._localize(answer, response_language)
            session.history.append({"role": "assistant", "content": localized_answer})
            return ChatResponse(
                session_id=session.session_id,
                status="needs_details",
                intent=analysis.intent,
                answer=localized_answer,
                english_answer=answer if response_language != "en-IN" else None,
                missing_fields=analysis.missing_fields,
                collected_details=session.collected_details,
                detected_language=response_language,
            )

        retrieval_message = self._rewrite_followup_for_retrieval(session, normalized_message)
        analysis.search_queries = self._default_search_queries(analysis.intent, session.collected_details, retrieval_message)
        rag_results: list[dict] = []
        search_results: list[dict] = []
        try:
            # Tavily-first pipeline: rely on live web search + model internal knowledge.
            # Do not depend on vector RAG retrieval for answer generation.
            search_results = self.search_service.search(analysis.search_queries)
            try:
                indexed_chunks = self.vector_store.add_search_results(search_results)
                logger.info("qdrant_indexed_chunks=%s search_results=%s", indexed_chunks, len(search_results))
            except Exception:
                logger.exception("qdrant_indexing_failed_non_blocking")
        except Exception:
            logger.exception("tavily_search_failed_non_blocking")
            search_results = []

        memory_hits = self.memory_service.search_relevant_feedback(session.user_id, retrieval_message)
        used_fallback = False
        try:
            model_answer = self._generate_answer(
                session=session,
                normalized_message=retrieval_message,
                search_results=search_results,
                rag_results=rag_results,
                memory_hits=memory_hits,
                output_language_code=response_language,
            )
            model_answer = self._clean_answer(model_answer)
        except Exception:
            used_fallback = True
            model_answer = self._build_fallback_answer(
                session=session,
                normalized_message=normalized_message,
                search_results=search_results,
                rag_results=rag_results,
            )
        if self._is_low_quality_answer(model_answer):
            used_fallback = True
            model_answer = self._build_fallback_answer(
                session=session,
                normalized_message=normalized_message,
                search_results=search_results,
                rag_results=rag_results,
            )
        should_force_localize = (
            response_language != "en-IN"
            and not self._looks_like_target_language(model_answer, response_language)
        )
        if used_fallback or should_force_localize:
            localized_answer = self._localize(model_answer, response_language)
        else:
            localized_answer = model_answer
        session.history.append({"role": "assistant", "content": localized_answer})

        source_map: dict[str, SourceItem] = {}
        for item in search_results + rag_results:
            url = item.get("url", "")
            if not url or url in source_map:
                continue
            source_map[url] = SourceItem(
                title=item.get("title", "Official source"),
                url=url,
                score=item.get("score"),
            )

        return ChatResponse(
            session_id=session.session_id,
            status="completed",
            intent=analysis.intent,
            answer=localized_answer,
            english_answer=model_answer if response_language != "en-IN" else None,
            collected_details=session.collected_details,
            sources=list(source_map.values())[:5],
            detected_language=response_language,
        )

    def _search_rag(self, queries: list[str], limit: int = 5) -> list[dict]:
        merged: list[dict] = []
        seen_urls: set[str] = set()
        for query in queries[:3]:
            for item in self.vector_store.search(query, limit=limit):
                url = item.get("url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                merged.append(item)
                if len(merged) >= limit:
                    return merged
        return merged

    def _build_retrieval_queries(self, normalized_message: str, details: dict[str, str], intent: str = "general") -> list[str]:
        queries = [normalized_message]
        state = details.get("state", "").strip()
        age = details.get("age", "").strip()
        profile = details.get("profile", "").strip()
        scheme = details.get("scheme_name", details.get("scheme_name_or_service", "")).strip()

        if intent in {"scheme_discovery", "eligibility_check"} and scheme:
            queries.append(f"{scheme} eligibility {state} {age} {profile}".strip())
        if intent in {"scheme_discovery", "eligibility_check"} and (state or age or profile):
            queries.append(f"government scheme {profile} age {age} state {state}".strip())
        if intent == "grievance_redressal" and scheme:
            queries.append(f"{scheme} grievance redressal portal {state}".strip())
        return [q for q in queries if q]

    def _analyze(
        self,
        session: SessionState,
        normalized_message: str,
        original_message: str | None = None,
    ) -> ConversationAnalysis:
        return self._fallback_analysis(session, normalized_message, original_message)

    def _generate_answer(
        self,
        *,
        session: SessionState,
        normalized_message: str,
        search_results: list[dict],
        rag_results: list[dict],
        memory_hits: list[str],
        output_language_code: str,
    ) -> str:
        retrieved_context_parts = []
        for item in search_results[:4]:
            retrieved_context_parts.append(
                f"Title: {item.get('title', 'Official source')}\nURL: {item.get('url', '')}\nContent: {item.get('content') or item.get('raw_content') or ''}"
            )
        for item in rag_results[:4]:
            retrieved_context_parts.append(
                f"RAG Title: {item.get('title', 'Stored source')}\nURL: {item.get('url', '')}\nContent: {item.get('content', '')}"
            )

        system_prompt = FINAL_RESPONSE_PROMPT.format(
            user_context=json.dumps(
                {
                    "intent": session.intent,
                    "details": session.collected_details,
                },
                ensure_ascii=True,
            ),
            memory_context="\n\n".join(memory_hits) if memory_hits else "No prior feedback available.",
            retrieved_context="\n\n".join(retrieved_context_parts) if retrieved_context_parts else "No trusted evidence retrieved.",
        )
        history_context = self._recent_history_context(session)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Recent conversation context:\n{history_context}\n\n"
                    f"User request:\n{normalized_message}\n\n"
                    f"Return a grounded answer in markdown.\n"
                    f"Use Tavily evidence first. If evidence is missing, use internal policy knowledge and clearly mark it as best-effort.\n"
                    f"Output language must be: {self._language_name(output_language_code)}."
                ),
            },
        ]
        return self.sarvam.chat_messages(messages, temperature=0.2)

    def _localize(self, answer: str, target_language_code: str) -> str:
        normalized_target = self._normalize_language_code(target_language_code)
        if normalized_target == "en-IN":
            return answer
        if self._looks_like_target_language(answer, normalized_target):
            return answer
        try:
            translated, _ = self.sarvam.translate(answer, "en-IN", normalized_target)
            if translated and translated.strip() and self._looks_like_target_language(translated, normalized_target):
                return translated.strip()

            translated_auto, _ = self.sarvam.translate(answer, "auto", normalized_target)
            if translated_auto and translated_auto.strip() and self._looks_like_target_language(translated_auto, normalized_target):
                return translated_auto.strip()

            # Backup path: force translation through chat channel when translate endpoint is flaky.
            fallback = self.sarvam.chat_messages(
                [
                    {
                        "role": "system",
                        "content": (
                            "Translate the text to the requested language only. "
                            "Do not add or remove information. Preserve markdown structure."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Target language: {self._language_name(normalized_target)}\n\nText:\n{answer}",
                    },
                ],
                temperature=0.0,
            )
            if isinstance(fallback, str) and fallback.strip() and self._looks_like_target_language(fallback, normalized_target):
                return fallback.strip()

            # Last resort: return any non-empty translation attempt over English.
            if translated_auto and translated_auto.strip() and translated_auto.strip() != answer.strip():
                return translated_auto.strip()
            if translated and translated.strip() and translated.strip() != answer.strip():
                return translated.strip()
            return answer
        except Exception:
            try:
                fallback = self.sarvam.chat_messages(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Translate the text to the requested language only. "
                                "Do not add commentary. Preserve markdown."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Target language: {self._language_name(normalized_target)}\n\nText:\n{answer}",
                        },
                    ],
                    temperature=0.0,
                )
                if isinstance(fallback, str) and fallback.strip():
                    return fallback.strip()
            except Exception:
                pass
            return answer

    def _clean_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        cleaned = re.sub(
            r"(?is)^.*?(?:1\.\s*Analyze the User's Request:|Analyze the User's Request:)",
            "",
            cleaned,
        ).strip()

        forbidden_patterns = [
            r"(?im)^\s*[-*]?\s*Core Need:.*$",
            r"(?im)^\s*[-*]?\s*User Context:.*$",
            r"(?im)^\s*[-*]?\s*Intent:.*$",
            r"(?im)^\s*[-*]?\s*Implicit Need:.*$",
        ]
        for pattern in forbidden_patterns:
            cleaned = re.sub(pattern, "", cleaned)

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned or answer.strip()

    def _is_low_quality_answer(self, answer: str) -> bool:
        text = answer.strip()
        if not text:
            return True

        letter_chars = len(re.findall(r"[^\W\d_]", text, flags=re.UNICODE))
        if letter_chars < 25:
            return True

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        non_link_lines = [line for line in lines if not re.search(r"https?://|^\[.*\]\(.*\)$", line)]
        if len(non_link_lines) < 2:
            return True

        link_count = len(re.findall(r"https?://|\[.*?\]\(.*?\)", text))
        if link_count >= 4 and len(non_link_lines) <= 2:
            return True

        return False

    def _build_fallback_answer(
        self,
        *,
        session: SessionState,
        normalized_message: str,
        search_results: list[dict],
        rag_results: list[dict],
    ) -> str:
        details = session.collected_details
        age = details.get("age", "the provided")
        profile = details.get("profile", "citizen")
        state = details.get("state", "your state")

        candidates = rag_results or search_results
        top = candidates[:2]

        summary_lines = [
            f"Based on your profile ({profile}), age ({age}), and location ({state}), here is a practical summary from official sources.",
        ]

        if top:
            summary_lines.append("")
            summary_lines.append("### Most Relevant Options")
            for item in top:
                title = item.get("title", "Official source")
                snippet = (item.get("content", "") or item.get("raw_content", "") or "").strip()
                snippet = re.sub(r"\s+", " ", snippet)
                snippet = snippet[:220].strip()
                if snippet:
                    summary_lines.append(f"- **{title}**: {snippet}")
                else:
                    summary_lines.append(f"- **{title}**: Please review the linked official source for exact eligibility details.")
        else:
            summary_lines.append("")
            summary_lines.append("### Current Status")
            summary_lines.append("- I could not find enough grounded policy text in the current cache for this request.")

        summary_lines.append("")
        summary_lines.append("### Next Steps")
        summary_lines.append("- Check the official scheme pages in the source links below for exact eligibility and required documents.")
        summary_lines.append("- Keep your age proof, residence proof, and occupation-related proof ready before applying.")
        summary_lines.append("- If one scheme does not fit, ask for alternatives and I will compare 2-3 options for your profile.")
        summary_lines.append("")
        summary_lines.append("_Note: Final eligibility is decided by the official department portal and latest notification._")
        return "\n".join(summary_lines)

    def _safe_detect_language(self, message: str) -> str:
        try:
            detected = self.sarvam.detect_language(message)
            return self._normalize_language_code(detected)
        except Exception:
            return self._infer_script_language(message)

    def _resolve_response_language(self, session: SessionState, user_message: str, language_code: str) -> str:
        requested_language = self._normalize_language_code(language_code) if language_code and language_code != "auto" else None
        explicit_language = self._infer_explicit_response_language(user_message)
        if explicit_language:
            return explicit_language

        # Always honor obvious script from the current user message.
        script_language = self._infer_script_language(user_message)
        if script_language != "en-IN":
            return script_language

        # If UI language is explicitly selected, lock to that for this turn.
        if requested_language:
            return requested_language

        # Session language lock: when language is auto and no explicit language-switch is requested,
        # keep replying in the previously established session language.
        if session.response_language:
            return session.response_language

        inferred_from_message = self._infer_message_language(user_message)
        if inferred_from_message:
            if (
                inferred_from_message == "en-IN"
                and session.response_language != "en-IN"
                and self._is_short_followup(user_message)
            ):
                return session.response_language
            return inferred_from_message

        detected_language = self._safe_detect_language(user_message)
        if detected_language:
            return detected_language

        return "en-IN"

    def _infer_explicit_response_language(self, message: str) -> str | None:
        lowered = message.lower()
        for language_code, hints in self.RESPONSE_LANGUAGE_HINTS.items():
            for hint in hints:
                if hint and hint.lower() in lowered:
                    return language_code
        return None

    def _is_short_followup(self, message: str) -> bool:
        cleaned = re.sub(r"[^\w\s]", " ", (message or "").lower())
        tokens = [token for token in cleaned.split() if token]
        if not tokens:
            return False
        if len(tokens) <= 5:
            return True
        followup_starters = {"and", "also", "then", "next", "what", "how", "where", "when", "why"}
        return len(tokens) <= 9 and tokens[0] in followup_starters

    def _recent_history_context(self, session: SessionState) -> str:
        if not session.history:
            return "No prior turns."
        recent_turns = session.history[-6:]
        lines: list[str] = []
        for item in recent_turns:
            role = item.get("role", "user")
            content = (item.get("content", "") or "").strip()
            if not content:
                continue
            condensed = re.sub(r"\s+", " ", content)[:320]
            lines.append(f"{role}: {condensed}")
        return "\n".join(lines) if lines else "No prior turns."

    def _fallback_analysis(
        self,
        session: SessionState,
        normalized_message: str,
        original_message: str | None = None,
    ) -> ConversationAnalysis:
        details = session.collected_details.copy()
        if self._looks_like_new_topic(normalized_message):
            details = self._prune_stale_details_for_new_topic(details, normalized_message)
        if original_message:
            details.update(self._extract_structured_details(original_message))
        details.update(self._extract_structured_details(normalized_message))
        intent = self._infer_intent(normalized_message, details)
        if (
            self._is_contextual_followup(normalized_message)
            and not self._looks_like_new_topic(normalized_message)
            and session.intent in {
            "scheme_discovery",
            "eligibility_check",
            "grievance_redressal",
            }
        ):
            intent = session.intent

        analysis = ConversationAnalysis(
            intent=intent,
            user_goal=normalized_message,
            collected_details=details,
            missing_fields=self._enforce_minimum_fields(intent, details),
            ready_for_search=False,
            clarifying_question="",
            search_queries=[],
        )
        analysis.ready_for_search = not analysis.missing_fields
        if analysis.ready_for_search:
            analysis.search_queries = self._default_search_queries(intent, details, normalized_message)
        else:
            analysis.clarifying_question = self._build_clarifying_question(analysis.missing_fields)
        return analysis

    def _is_contextual_followup(self, message: str) -> bool:
        cleaned = re.sub(r"\s+", " ", (message or "").strip().lower())
        if not cleaned:
            return False
        if len(cleaned.split()) <= 7:
            return True
        followup_phrases = [
            "what about",
            "and for",
            "and also",
            "how about",
            "documents required",
            "required documents",
            "eligibility for this",
            "for this scheme",
            "for this one",
            "apply for this",
            "when will i get",
            "status of this",
        ]
        return any(phrase in cleaned for phrase in followup_phrases)

    def _looks_like_new_topic(self, message: str) -> bool:
        cleaned = re.sub(r"\s+", " ", (message or "").strip().lower())
        if not cleaned:
            return False
        topic_markers = [
            "scheme",
            "yojana",
            "scholarship",
            "certificate",
            "income certificate",
            "domicile",
            "caste certificate",
            "birth certificate",
            "death certificate",
            "ration card",
            "aadhar",
            "aadhaar",
            "voter id",
            "pan card",
            "pension",
            "benefit",
            "apply for",
            "eligibility",
            "documents",
            "grievance",
            "complaint",
        ]
        return any(marker in cleaned for marker in topic_markers)

    def _prune_stale_details_for_new_topic(self, details: dict[str, str], message: str) -> dict[str, str]:
        cleaned = (message or "").lower()
        kept: dict[str, str] = {}
        # Keep only generally reusable user attributes.
        for key in ("state", "age", "profile"):
            if details.get(key):
                kept[key] = details[key]

        # If user still clearly talks about grievance, keep grievance context.
        if any(word in cleaned for word in ("grievance", "complaint", "issue", "delay")):
            for key in ("scheme_name_or_service", "grievance_summary"):
                if details.get(key):
                    kept[key] = details[key]
        return kept

    def _rewrite_followup_for_retrieval(self, session: SessionState, message: str) -> str:
        cleaned = re.sub(r"\s+", " ", (message or "").strip())
        if not cleaned:
            return message
        if self._looks_like_new_topic(cleaned):
            return cleaned
        if not self._is_contextual_followup(cleaned):
            return cleaned

        previous_user_turn = self._previous_user_turn(session)
        details = session.collected_details
        detail_chunks: list[str] = []
        if details.get("scheme_name"):
            detail_chunks.append(f"scheme: {details['scheme_name']}")
        if details.get("scheme_name_or_service"):
            detail_chunks.append(f"service: {details['scheme_name_or_service']}")
        if details.get("state"):
            detail_chunks.append(f"state: {details['state']}")
        if details.get("age"):
            detail_chunks.append(f"age: {details['age']}")
        if details.get("profile"):
            detail_chunks.append(f"profile: {details['profile']}")
        if details.get("grievance_summary"):
            detail_chunks.append(f"grievance: {details['grievance_summary'][:120]}")

        intent = session.intent if session.intent in {
            "scheme_discovery",
            "eligibility_check",
            "grievance_redressal",
            "general",
        } else "general"
        context_line = f"intent: {intent}"
        if detail_chunks:
            context_line = f"{context_line}; " + "; ".join(detail_chunks)

        if previous_user_turn:
            return (
                f"Follow-up question with context. Previous user request: {previous_user_turn}. "
                f"Current follow-up: {cleaned}. Context: {context_line}."
            )
        return f"Follow-up question: {cleaned}. Context: {context_line}."

    def _previous_user_turn(self, session: SessionState) -> str:
        if not session.history:
            return ""
        # Ignore current user turn (already appended) and fetch previous user message.
        for item in reversed(session.history[:-1]):
            if item.get("role") != "user":
                continue
            content = re.sub(r"\s+", " ", (item.get("content", "") or "").strip())
            if content:
                return content[:320]
        return ""

    def _should_search_web(self, rag_results: list[dict]) -> bool:
        if not rag_results:
            return True
        top_score = rag_results[0].get("score")
        if top_score is None:
            return len(rag_results) < 2
        return len(rag_results) < 2 or top_score < 0.45

    def _should_refresh_index(self, retrieval_message: str, rag_results: list[dict]) -> bool:
        # Even when RAG has results, fetch/index fresh sources for likely new topics
        # so vector store keeps growing with current policy pages.
        if not self._looks_like_new_topic(retrieval_message):
            return False
        if not rag_results:
            return True

        query_tokens = self._important_tokens(retrieval_message)
        if not query_tokens:
            return False

        rag_text = " ".join(
            f"{item.get('title', '')} {item.get('content', '')}" for item in rag_results[:3]
        ).lower()
        matched = sum(1 for token in query_tokens if token in rag_text)
        # Refresh when overlap is weak, indicating current index may not have the new policy.
        return matched < max(1, len(query_tokens) // 3)

    def _important_tokens(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", (text or "").lower())
        stopwords = {
            "what",
            "which",
            "about",
            "please",
            "tell",
            "need",
            "want",
            "details",
            "information",
            "scheme",
            "yojana",
            "policy",
            "government",
        }
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in stopwords:
                continue
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return deduped[:8]

    def _normalize_language_code(self, language_code: str | None) -> str:
        if not language_code:
            return "en-IN"
        normalized = language_code.strip().lower()
        return self.LANGUAGE_CODE_MAP.get(normalized, language_code)

    def _language_name(self, language_code: str) -> str:
        normalized = self._normalize_language_code(language_code)
        return self.LANGUAGE_NAME_MAP.get(normalized, "English")

    def _looks_like_target_language(self, text: str, language_code: str) -> bool:
        if not text.strip():
            return False
        normalized = self._normalize_language_code(language_code)
        if normalized == "en-IN":
            latin_count = len(re.findall(r"[A-Za-z]", text))
            return latin_count >= 20

        script_checks = {
            "hi-IN": r"[\u0900-\u097F]",
            "ta-IN": r"[\u0B80-\u0BFF]",
            "te-IN": r"[\u0C00-\u0C7F]",
            "kn-IN": r"[\u0C80-\u0CFF]",
            "ml-IN": r"[\u0D00-\u0D7F]",
            "bn-IN": r"[\u0980-\u09FF]",
            "pa-IN": r"[\u0A00-\u0A7F]",
            "gu-IN": r"[\u0A80-\u0AFF]",
            "od-IN": r"[\u0B00-\u0B7F]",
        }
        pattern = script_checks.get(normalized)
        if not pattern:
            return True
        return bool(re.search(pattern, text))

    def _infer_script_language(self, message: str) -> str:
        if re.search(r"[\u0900-\u097F]", message):
            return "hi-IN"
        if re.search(r"[\u0B80-\u0BFF]", message):
            return "ta-IN"
        if re.search(r"[\u0C00-\u0C7F]", message):
            return "te-IN"
        if re.search(r"[\u0C80-\u0CFF]", message):
            return "kn-IN"
        if re.search(r"[\u0D00-\u0D7F]", message):
            return "ml-IN"
        if re.search(r"[\u0980-\u09FF]", message):
            return "bn-IN"
        if re.search(r"[\u0A00-\u0A7F]", message):
            return "pa-IN"
        if re.search(r"[\u0A80-\u0AFF]", message):
            return "gu-IN"
        return "en-IN"

    def _infer_message_language(self, message: str) -> str | None:
        script_language = self._infer_script_language(message)
        if script_language != "en-IN":
            return script_language

        latin_letters = len(re.findall(r"[A-Za-z]", message))
        non_latin_indic = len(re.findall(r"[\u0900-\u0D7F]", message))
        if latin_letters > 0 and non_latin_indic == 0:
            return "en-IN"

        return None

    def _extract_structured_details(self, message: str) -> dict[str, str]:
        message = self._normalize_numerals(message)
        lowered = message.lower()
        details: dict[str, str] = {}

        age_patterns = [
            r"\b(\d{1,3})\s*[- ]?years?[- ]?old\b",
            r"\baged?\s*(?:is\s*)?(\d{1,3})\b",
            r"\bage\s*(?:is|=|:)?\s*(\d{1,3})\b",
            r"\bi am\s+(\d{1,3})\b",
            r"\bmy\s+(?:mother|father|mom|dad|wife|husband|son|daughter)\s+is\s+(\d{1,3})\b",
            r"आयु\s*(?:है|is)?\s*(\d{1,2})",
            r"उम्र\s*(?:है|is)?\s*(\d{1,2})",
        ]
        for pattern in age_patterns:
            match = re.search(pattern, lowered if "\\b" in pattern or "age" in pattern or "year" in pattern else message)
            if match:
                candidate = match.group(1).strip()
                try:
                    numeric_age = int(candidate)
                except ValueError:
                    continue
                if 0 < numeric_age <= 120:
                    details["age"] = str(numeric_age)
                    break

        # Accept concise follow-ups like "45" when user is filling missing age.
        if not details.get("age"):
            standalone_age = re.fullmatch(r"\s*(\d{1,3})\s*", lowered)
            if standalone_age:
                numeric_age = int(standalone_age.group(1))
                if 0 < numeric_age <= 120:
                    details["age"] = str(numeric_age)

        for state_key, state_value in sorted(self.STATE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\bfrom\s+{re.escape(state_key)}\b", lowered) or re.search(
                rf"\bin\s+{re.escape(state_key)}\b", lowered
            ) or re.search(
                rf"\bstate\s*(?:is|=)?\s*{re.escape(state_key)}\b", lowered
            ):
                details["state"] = state_value
                break

        # Accept concise follow-ups like "Karnataka" when user is filling missing fields.
        if not details.get("state"):
            compact = re.sub(r"[^a-z\s]", " ", lowered)
            compact = re.sub(r"\s+", " ", compact).strip()
            for state_key, state_value in sorted(self.STATE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
                if re.fullmatch(rf"{re.escape(state_key)}", compact) or re.search(rf"\b{re.escape(state_key)}\b", compact):
                    details["state"] = state_value
                    break

        if not details.get("state"):
            for state_key, state_value in sorted(self.STATE_ALIASES_HI.items(), key=lambda item: len(item[0]), reverse=True):
                if (
                    state_key in message
                    or re.search(rf"राज्य\s+{re.escape(state_key)}", message)
                    or re.search(rf"{re.escape(state_key)}\s+राज्य", message)
                ):
                    details["state"] = state_value
                    break

        for profile_key, profile_value in sorted(self.PROFILE_KEYWORDS.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(profile_key)}\b", lowered):
                details["profile"] = profile_value
                break

        # Fallback: capture common occupation phrases even if exact keyword is missing.
        if not details.get("profile"):
            occupation_match = re.search(
                r"\b(?:i am|i'm|working as|occupation is)\s+(?:a|an)?\s*([a-z][a-z ]{2,40})\b",
                lowered,
            )
            if occupation_match:
                candidate = occupation_match.group(1).strip()
                candidate = re.sub(r"\b(in|from|at)\b.*$", "", candidate).strip()
                if candidate:
                    details["profile"] = candidate

        # Handle explicit profile-style statements, e.g. "profile is lawyer".
        if not details.get("profile"):
            profile_match = re.search(
                r"\bprofile\s*(?:is|=|:)\s*(?:a|an)?\s*([a-z][a-z ]{2,40})\b",
                lowered,
            )
            if profile_match:
                candidate = profile_match.group(1).strip()
                candidate = re.sub(r"\b(in|from|at)\b.*$", "", candidate).strip()
                if candidate:
                    details["profile"] = candidate

        scheme_match = re.search(
            r"(?:scheme|yojana|program|programme)\s+(?:called\s+|named\s+)?([A-Z][A-Za-z0-9 .'-]{2,60})",
            message,
        )
        if scheme_match and "what schemes" not in lowered and "which schemes" not in lowered:
            details["scheme_name"] = scheme_match.group(1).strip(" .")
        if not details.get("scheme_name"):
            scheme_reverse_match = re.search(
                r"\b([A-Za-z][A-Za-z0-9 .'-]{1,80}?)\s+(?:scheme|yojana|program|programme)\b",
                message,
                flags=re.IGNORECASE,
            )
            if scheme_reverse_match and "what schemes" not in lowered and "which schemes" not in lowered:
                candidate = scheme_reverse_match.group(1).strip(" .")
                if candidate:
                    details["scheme_name"] = f"{candidate} scheme"

        if any(keyword in lowered for keyword in ["grievance", "complaint", "issue", "not received", "delay", "problem"]):
            details["grievance_summary"] = message.strip()
            service_match = re.search(r"(?:for|regarding|about)\s+([A-Za-z][A-Za-z0-9 .'-]{2,60})", message)
            if service_match:
                details["scheme_name_or_service"] = service_match.group(1).strip(" .")

        return details

    def _normalize_numerals(self, message: str) -> str:
        devanagari_digits = str.maketrans("०१२३४५६७८९", "0123456789")
        return message.translate(devanagari_digits)

    def _infer_intent(self, message: str, details: dict[str, str]) -> str:
        lowered = message.lower()
        if self._is_document_service_query(lowered):
            return "general"

        if any(keyword in lowered for keyword in ["grievance", "complaint", "issue", "not received", "delay", "problem"]):
            return "grievance_redressal"

        scheme_discovery_patterns = [
            "what schemes",
            "which schemes",
            "government schemes",
            "any schemes",
            "schemes can i apply for",
            "schemes am i eligible for",
            "find schemes",
        ]
        if any(pattern in lowered for pattern in scheme_discovery_patterns):
            return "scheme_discovery"

        if details.get("scheme_name") and any(
            keyword in lowered for keyword in ["eligible", "eligibility", "can i apply", "am i eligible", "qualify"]
        ):
            return "eligibility_check"

        if details.get("scheme_name"):
            return "eligibility_check"

        return "scheme_discovery"

    def _is_document_service_query(self, lowered_message: str) -> bool:
        patterns = [
            "income certificate",
            "caste certificate",
            "domicile certificate",
            "residence certificate",
            "birth certificate",
            "death certificate",
            "marriage certificate",
            "ration card",
            "aadhar",
            "aadhaar",
            "voter id",
            "pan card",
        ]
        return any(pattern in lowered_message for pattern in patterns)

    def _resolve_intent(self, *, llm_intent: str, heuristic_intent: str, message: str, details: dict[str, str]) -> str:
        lowered = message.lower()
        if any(pattern in lowered for pattern in ["what schemes", "which schemes", "any schemes", "schemes can i apply for"]):
            return "scheme_discovery"
        if heuristic_intent == "eligibility_check" and not details.get("scheme_name"):
            return "scheme_discovery"
        if llm_intent == "eligibility_check" and not details.get("scheme_name"):
            return "scheme_discovery"
        return heuristic_intent or llm_intent

    def _enforce_minimum_fields(self, intent: str, details: dict[str, str]) -> list[str]:
        required_map = {
            "scheme_discovery": ["state", "age", "profile"],
            "eligibility_check": ["scheme_name", "state", "age", "profile"],
            "grievance_redressal": ["scheme_name_or_service", "state", "grievance_summary"],
            "general": [],
        }
        required_fields = required_map.get(intent, [])
        return [field for field in required_fields if not details.get(field)]

    def _build_clarifying_question(self, missing_fields: list[str]) -> str:
        readable = ", ".join(field.replace("_", " ") for field in missing_fields)
        return f"To help accurately, please share your {readable}."

    def _default_search_queries(self, intent: str, details: dict[str, str], normalized_message: str) -> list[str]:
        state = details.get("state", "")
        age = details.get("age", "")
        profile = details.get("profile", "")
        scheme_name = details.get("scheme_name", details.get("scheme_name_or_service", ""))
        grievance = details.get("grievance_summary", "")

        if intent == "eligibility_check":
            return [
                f"{scheme_name} eligibility criteria {state} {age} {profile} site:gov.in",
                f"{scheme_name} official guidelines {state} site:gov.in",
            ]
        if intent == "grievance_redressal":
            return [
                f"{scheme_name} grievance redressal {state} {grievance} site:gov.in",
                f"{scheme_name} complaint portal {state} site:gov.in",
            ]
        return [
            f"government schemes for {profile} age {age} in {state} site:gov.in",
            f"{state} official welfare schemes for {profile} site:gov.in",
            normalized_message,
        ]
