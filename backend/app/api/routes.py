from __future__ import annotations

import logging

import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from backend.app.core.security import contains_prompt_injection, sanitize_user_text
from backend.app.models.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
)


router = APIRouter()
logger = logging.getLogger(__name__)


def get_app_state(request: Request):
    return request.app.state


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request, state=Depends(get_app_state)) -> ChatResponse:
    client_ip = request.client.host if request.client else "unknown"
    if not state.rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down and try again.")

    user_message = sanitize_user_text(payload.message)
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    guardrail_triggered = contains_prompt_injection(user_message)
    if guardrail_triggered:
        return ChatResponse(
            session_id=payload.session_id or "guardrail-blocked",
            status="blocked",
            intent="general",
            answer="I can help with public-service queries, but I cannot follow instructions that try to override system safeguards. Please restate your request in plain terms.",
            detected_language="en-IN",
            guardrail_triggered=True,
        )

    session = state.session_store.get_or_create(payload.session_id, payload.user_id)
    try:
        response = state.policy_agent.handle_message(session, user_message, payload.language_code)
        response.guardrail_triggered = False
        state.session_store.save(session)
        return response
    except Exception:
        logger.exception("chat_processing_failed session_id=%s", session.session_id)
        return ChatResponse(
            session_id=session.session_id,
            status="blocked",
            intent=session.intent if session.intent in {"scheme_discovery", "eligibility_check", "grievance_redressal", "general"} else "general",
            answer="I hit a temporary backend issue while processing this message. Please try again once. If it repeats, check server logs for `chat_processing_failed`.",
            detected_language=session.response_language or session.detected_language or "en-IN",
            collected_details=session.collected_details,
            guardrail_triggered=False,
        )


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest, request: Request, state=Depends(get_app_state)) -> FeedbackResponse:
    session = state.session_store.get_or_create(payload.session_id, payload.user_id)
    stored_with_mem0 = state.memory_service.remember_feedback(
        user_id=session.user_id,
        helpful=payload.helpful,
        feedback_text=payload.feedback_text or "",
        answer_snapshot=payload.answer_snapshot,
    )
    return FeedbackResponse(
        success=True,
        stored_with_mem0=stored_with_mem0,
        message="Feedback stored successfully.",
    )


@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(
    request: Request,
    state=Depends(get_app_state),
    file: UploadFile = File(...),
    language_code: str = Form(default="auto"),
) -> SpeechToTextResponse:
    client_ip = request.client.host if request.client else "unknown"
    if not state.rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down and try again.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    transcript, detected_language = state.policy_agent.sarvam.speech_to_text(
        file_bytes=audio_bytes,
        filename=file.filename or "audio.webm",
        content_type=file.content_type or "audio/webm",
        language_code=language_code,
    )
    if not transcript:
        raise HTTPException(status_code=422, detail="Could not transcribe the uploaded audio.")

    return SpeechToTextResponse(
        transcript=transcript,
        detected_language=detected_language,
    )


@router.post("/text-to-speech", response_model=TextToSpeechResponse)
def text_to_speech(payload: TextToSpeechRequest, request: Request, state=Depends(get_app_state)) -> TextToSpeechResponse:
    client_ip = request.client.host if request.client else "unknown"
    if not state.rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down and try again.")

    try:
        audio_base64, audio_mime_type = state.policy_agent.sarvam.text_to_speech(
            text=payload.text,
            language_code=payload.language_code,
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        if status == 429:
            raise HTTPException(
                status_code=429,
                detail="Text-to-speech rate limit reached. Please retry after a short delay.",
            ) from exc
        logger.exception("tts_http_error status=%s", status)
        raise HTTPException(status_code=502, detail="Text-to-speech service returned an error.") from exc
    except Exception as exc:
        logger.exception("tts_unexpected_error")
        raise HTTPException(status_code=502, detail="Text-to-speech failed due to a backend error.") from exc

    if not audio_base64:
        raise HTTPException(status_code=502, detail="Text-to-speech generation failed.")

    return TextToSpeechResponse(
        audio_base64=audio_base64,
        audio_mime_type=audio_mime_type,
        language_code=payload.language_code,
    )


@router.get("/debug/vector-status")
def vector_status(request: Request, state=Depends(get_app_state)) -> dict:
    stats = state.policy_agent.vector_store.get_collection_stats()
    return {
        "collection_name": state.policy_agent.vector_store.collection_name,
        "embedding_model": state.policy_agent.vector_store.embedding_model,
        **stats,
    }
