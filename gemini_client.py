from __future__ import annotations

import json
import os
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_key = os.getenv("GOOGLE_API_KEY")

SYSTEM_INSTRUCTION = (
    "You are a Requirement Engineer. Your goal is to gather details to create "
    "an optimized AI prompt. Ask one question at a time. If you have enough info, "
    "provide a summary and then the final prompt."
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
    api_key = API_key
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set")

    genai.configure(api_key=api_key)
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
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Gemini returned non-object JSON")
    return _validate_response(payload)
