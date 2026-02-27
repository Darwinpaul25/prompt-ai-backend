from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))


def _get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
    return api_key

SYSTEM_INSTRUCTION = (
    """
    Role: You are "Prompt Buddy," a legendary AI architect and the ultimate hype-man for creative ideas! You are an expert in Prompt Engineering and Requirement Elicitation.
    The Vibe: > * Be Energetic: Use phrases like "Let's go!", "That's a killer idea!", and "We're going to build something epic today! 🚀"
    Be Supportive: If the user is vague, do't be a robot. Say something like, "Ooh, I see where you're going with this! Help me sharpen the vision..."
    Use Emojis: Strategically use 1-2 emojis per response to keep the mood light and fun.
    The Workflow:
    Phase 1 (The Hook): Celebrate the user's initial goal and ask the first punchy question.
    Phase 2 (The Deep Dive): Ask exactly ONE question at a time. Use UI elements to make it easy for them.
    Phase 3 (The Recap): Once you have enough info, say "Alright, let's look at the blueprint! 🛠️" and summarize the requirements.
    Phase 4 (The Reveal): Deliver the "God-tier" optimized prompt that they can paste elsewhere.
    """
)

JSON_RULES = (
    "Return ONLY strict JSON. No markdown, no extra keys, no commentary.\n"
    "Required schema:\n"
    '{\n'
    '  "status": "collecting" | "delivered",\n'
    '  "question_text": "string",\n'
    '  "ui_elements": [\n'
    "    {\n"
    '      "type": "radio" | "checkbox" | "text",\n'
    '      "options": ["string", ...]\n'
    "    }\n"
    "  ],\n"
    '  "final_prompt": "string"\n'
    "}\n"
    "When type is radio, every option must start with '( ) '.\n"
    "When type is checkbox, every option must start with '[ ] '.\n"
    "When type is text, options must be []."
)


def _normalize_ui_elements(ui_elements: Any) -> list[dict[str, Any]]:
    if not isinstance(ui_elements, list):
        return []

    normalized: list[dict[str, Any]] = []
    for el in ui_elements:
        if not isinstance(el, dict):
            continue
        el_type = el.get("type")
        options = el.get("options", [])
        if not isinstance(options, list):
            options = []

        cleaned_options: list[str] = [str(opt).strip() for opt in options]
        if el_type == "radio":
            cleaned_options = [
                opt if opt.startswith("( ) ") else f"( ) {opt}" for opt in cleaned_options
            ]
        elif el_type == "checkbox":
            cleaned_options = [
                opt if opt.startswith("[ ] ") else f"[ ] {opt}" for opt in cleaned_options
            ]
        elif el_type == "text":
            cleaned_options = []
        else:
            continue

        normalized.append({"type": el_type, "options": cleaned_options})
    return normalized


def _validate_response(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if status not in {"collecting", "delivered"}:
        status = "collecting"

    question_text = payload.get("question_text")
    if not isinstance(question_text, str):
        question_text = ""

    final_prompt = payload.get("final_prompt")
    if not isinstance(final_prompt, str):
        final_prompt = ""

    ui_elements = _normalize_ui_elements(payload.get("ui_elements"))

    return {
        "status": status,
        "question_text": question_text,
        "ui_elements": ui_elements,
        "final_prompt": final_prompt,
    }


def get_gemini_response(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate the next requirement-engineering turn as strict JSON."""
    genai.configure(api_key=_get_api_key())
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
    )

    prompt_history = history if isinstance(history, list) else []
    response = model.generate_content(
        [*prompt_history, {"role": "user", "parts": [JSON_RULES]}],
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
        },
    )

    raw_text = (response.text or "").strip()
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback when the model wraps JSON with extra text.
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise ValueError("Gemini did not return valid JSON")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Gemini returned non-object JSON")
    return _validate_response(payload)


def get_session_title(user_input: str) -> str:
    """Generate a short (3-4 words) session title from user intent."""
    genai.configure(api_key=_get_api_key())
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    prompt = (
        "Summarize the user's intent in exactly 3-4 words. No punctuation.\n\n"
        f"User input: {user_input}"
    )
    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    raw_text = (response.text or "").strip()
    words = [re.sub(r"[^A-Za-z0-9]+", "", w) for w in raw_text.split()]
    words = [w for w in words if w]
    if len(words) >= 4:
        words = words[:4]
    elif len(words) == 0:
        words = ["New", "Prompt", "Session"]
    return " ".join(words)
