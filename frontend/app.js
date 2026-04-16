const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const chatLog = document.getElementById("chatLog");
const sendButton = document.getElementById("sendButton");
const languageSelect = document.getElementById("language");
const quickChips = document.querySelectorAll(".quick-chip");
const recordButton = document.getElementById("recordButton");
const autoSpeakToggle = document.getElementById("autoSpeakToggle");
const recentPoliciesList = document.getElementById("recentPoliciesList");
const clearRecentPoliciesBtn = document.getElementById("clearRecentPoliciesBtn");

let sessionId = crypto.randomUUID();
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let activeLoader = null;
let autoSpeakArmed = false;
const RECENT_POLICIES_KEY = "recentPoliciesV1";
const recentPolicies = loadRecentPolicies();

if (window.marked) {
  marked.setOptions({
    breaks: true,
    gfm: true,
  });
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderMessageContent(container, role, text) {
  if (role === "assistant" && window.marked) {
    container.innerHTML = marked.parse(escapeHtml(text));
    return;
  }

  container.textContent = text;
}

function addMessage(role, text, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const content = document.createElement("div");
  content.className = "message-content";
  renderMessageContent(content, role, text);
  bubble.appendChild(content);

  if (role === "assistant" && text?.trim()) {
    const ttsRow = document.createElement("div");
    ttsRow.className = "tts-row";
    const ttsButton = document.createElement("button");
    ttsButton.type = "button";
    ttsButton.className = "tts-btn";
    ttsButton.textContent = "Speak";
    ttsButton.addEventListener("click", async () => {
      ttsButton.disabled = true;
      const oldLabel = ttsButton.textContent;
      ttsButton.textContent = "Speaking...";
      try {
        await speakText(text, options.languageCode || languageSelect.value || "en-IN");
      } finally {
        ttsButton.textContent = oldLabel;
        ttsButton.disabled = false;
      }
    });
    ttsRow.appendChild(ttsButton);
    bubble.appendChild(ttsRow);

    const feedbackRow = document.createElement("div");
    feedbackRow.className = "feedback-row";

    const helpfulButton = document.createElement("button");
    helpfulButton.type = "button";
    helpfulButton.className = "feedback-btn";
    helpfulButton.textContent = "Helpful";

    const notHelpfulButton = document.createElement("button");
    notHelpfulButton.type = "button";
    notHelpfulButton.className = "feedback-btn";
    notHelpfulButton.textContent = "Needs Improvement";

    const setFeedbackDisabled = (disabled) => {
      helpfulButton.disabled = disabled;
      notHelpfulButton.disabled = disabled;
    };

    helpfulButton.addEventListener("click", async () => {
      setFeedbackDisabled(true);
      await submitFeedback(true, text, feedbackRow);
    });

    notHelpfulButton.addEventListener("click", async () => {
      setFeedbackDisabled(true);
      await submitFeedback(false, text, feedbackRow);
    });

    feedbackRow.appendChild(helpfulButton);
    feedbackRow.appendChild(notHelpfulButton);
    bubble.appendChild(feedbackRow);
  }

  article.appendChild(bubble);
  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addAssistantLoader() {
  removeAssistantLoader();
  const article = document.createElement("article");
  article.className = "message assistant loading-message";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `
    <div class="loader-wrap">
      <span class="loader-dot"></span>
      <span class="loader-dot"></span>
      <span class="loader-dot"></span>
      <span class="loader-text">Assistant is thinking...</span>
    </div>
  `;

  article.appendChild(bubble);
  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
  activeLoader = article;
}

function removeAssistantLoader() {
  if (activeLoader && activeLoader.parentNode) {
    activeLoader.parentNode.removeChild(activeLoader);
  }
  activeLoader = null;
}

function loadRecentPolicies() {
  try {
    const raw = window.localStorage.getItem(RECENT_POLICIES_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((item) => item && typeof item.title === "string" && typeof item.url === "string");
  } catch {
    return [];
  }
}

function saveRecentPolicies() {
  try {
    window.localStorage.setItem(RECENT_POLICIES_KEY, JSON.stringify(recentPolicies));
  } catch {
    // Ignore storage errors in private mode or quota limits.
  }
}

function renderRecentPolicies() {
  if (!recentPoliciesList) {
    return;
  }

  recentPoliciesList.innerHTML = "";
  if (!recentPolicies.length) {
    const empty = document.createElement("li");
    empty.className = "empty-policy-item";
    empty.textContent = "No policies viewed yet.";
    recentPoliciesList.appendChild(empty);
    return;
  }

  recentPolicies.forEach((item) => {
    const li = document.createElement("li");
    li.className = "policy-item";

    const link = document.createElement("a");
    link.className = "policy-link";
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = item.title || item.url;

    li.appendChild(link);
    recentPoliciesList.appendChild(li);
  });
}

function updateRecentPoliciesFromSources(sources) {
  if (!Array.isArray(sources) || !sources.length) {
    return;
  }

  for (const source of sources) {
    if (!source || typeof source.url !== "string" || !source.url.trim()) {
      continue;
    }
    const normalizedUrl = source.url.trim();
    const title = (source.title || "Policy Source").trim();

    const existingIndex = recentPolicies.findIndex((item) => item.url === normalizedUrl);
    if (existingIndex >= 0) {
      recentPolicies.splice(existingIndex, 1);
    }
    recentPolicies.unshift({ title, url: normalizedUrl });
  }

  if (recentPolicies.length > 12) {
    recentPolicies.length = 12;
  }
  saveRecentPolicies();
  renderRecentPolicies();
}

async function sendChatMessage(message) {
  if (!message) {
    return;
  }

  addMessage("user", message);
  messageInput.value = "";
  sendButton.disabled = true;
  addAssistantLoader();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        language_code: languageSelect.value,
      }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }
    removeAssistantLoader();

    sessionId = payload.session_id || sessionId;

    addMessage("assistant", payload.answer, {
      languageCode: payload.detected_language || languageSelect.value || "en-IN",
    });
    updateRecentPoliciesFromSources(payload.sources);

    if (autoSpeakToggle?.checked && autoSpeakArmed && payload.answer) {
      await speakText(payload.answer, payload.detected_language || languageSelect.value || "en-IN");
    }
  } catch (error) {
    removeAssistantLoader();
    addMessage(
      "assistant",
      "I could not process that request right now. Please check your backend configuration and try again."
    );
  } finally {
    removeAssistantLoader();
    sendButton.disabled = false;
  }
}

function base64ToBlob(base64, mimeType) {
  const byteChars = atob(base64);
  const byteNumbers = new Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i += 1) {
    byteNumbers[i] = byteChars.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: mimeType || "audio/wav" });
}

async function speakText(text, languageCode) {
  const response = await fetch("/api/text-to-speech", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      language_code: languageCode || "en-IN",
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "TTS failed");
  }
  const audioBlob = base64ToBlob(payload.audio_base64, payload.audio_mime_type);
  const audioUrl = URL.createObjectURL(audioBlob);
  const audio = new Audio(audioUrl);
  await audio.play();
  audio.onended = () => URL.revokeObjectURL(audioUrl);
}

async function transcribeAudio(blob) {
  const formData = new FormData();
  formData.append("file", blob, "voice-input.webm");
  formData.append("language_code", languageSelect.value || "auto");
  const response = await fetch("/api/speech-to-text", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Speech-to-text failed");
  }
  return payload.transcript;
}

function updateRecordButtonState() {
  if (!recordButton) {
    return;
  }
  recordButton.classList.toggle("recording", isRecording);
  recordButton.textContent = isRecording ? "Stop Voice Input" : "Start Voice Input";
}

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(recordedChunks, { type: "audio/webm" });
    recordedChunks = [];
    try {
      const transcript = await transcribeAudio(audioBlob);
      if (transcript) {
        messageInput.value = transcript;
        await sendChatMessage(transcript);
      }
    } catch (error) {
      addMessage("assistant", "Voice input could not be processed. Please try again.");
    }
  };
  mediaRecorder.start();
  isRecording = true;
  updateRecordButtonState();
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((track) => track.stop());
  }
  isRecording = false;
  updateRecordButtonState();
}

async function submitFeedback(helpful, answerSnapshot, row) {
  const feedbackText = window.prompt(
    helpful
      ? "Optional: what worked well?"
      : "Optional: what was missing or incorrect?"
  );

  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        helpful,
        feedback_text: feedbackText || "",
        answer_snapshot: answerSnapshot,
      }),
    });

    if (!response.ok) {
      throw new Error("Feedback request failed");
    }

    row.textContent = "Thanks. Your feedback has been stored.";
  } catch (error) {
    row.textContent = "Feedback could not be saved right now.";
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  if (!message) {
    return;
  }
  await sendChatMessage(message);
});

quickChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    const text = chip.getAttribute("data-fill");
    if (!text) {
      return;
    }
    messageInput.value = text;
    messageInput.focus();
  });
});

if (recordButton) {
  updateRecordButtonState();
  recordButton.addEventListener("click", async () => {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      addMessage("assistant", "Voice input is not supported in this browser.");
      return;
    }
    try {
      if (isRecording) {
        stopRecording();
      } else {
        await startRecording();
      }
    } catch (error) {
      isRecording = false;
      updateRecordButtonState();
      addMessage("assistant", "Microphone permission failed. Please allow mic access and try again.");
    }
  });
}

if (autoSpeakToggle) {
  autoSpeakToggle.checked = false;
  autoSpeakToggle.addEventListener("change", () => {
    autoSpeakArmed = autoSpeakToggle.checked;
  });
}

if (clearRecentPoliciesBtn) {
  clearRecentPoliciesBtn.addEventListener("click", () => {
    recentPolicies.length = 0;
    saveRecentPolicies();
    renderRecentPolicies();
  });
}

renderRecentPolicies();
