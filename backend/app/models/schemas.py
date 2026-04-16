from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IntentType = Literal["scheme_discovery", "eligibility_check", "grievance_redressal", "general"]
StatusType = Literal["needs_details", "completed", "blocked"]


class SourceItem(BaseModel):
    title: str
    url: str
    score: float | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    user_id: str | None = None
    language_code: str = "auto"


class ChatResponse(BaseModel):
    session_id: str
    status: StatusType
    intent: IntentType
    answer: str
    english_answer: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    collected_details: dict[str, str] = Field(default_factory=dict)
    sources: list[SourceItem] = Field(default_factory=list)
    detected_language: str = "en-IN"
    guardrail_triggered: bool = False


class FeedbackRequest(BaseModel):
    session_id: str
    user_id: str | None = None
    helpful: bool
    feedback_text: str | None = Field(default=None, max_length=500)
    answer_snapshot: str | None = Field(default=None, max_length=4000)


class FeedbackResponse(BaseModel):
    success: bool
    stored_with_mem0: bool = False
    message: str


class SpeechToTextResponse(BaseModel):
    transcript: str
    detected_language: str = "en-IN"


class TextToSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    language_code: str = "en-IN"


class TextToSpeechResponse(BaseModel):
    audio_base64: str
    audio_mime_type: str = "audio/wav"
    language_code: str = "en-IN"


class ConversationAnalysis(BaseModel):
    intent: IntentType = "general"
    user_goal: str = ""
    collected_details: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    ready_for_search: bool = False
    clarifying_question: str = ""
    search_queries: list[str] = Field(default_factory=list)
