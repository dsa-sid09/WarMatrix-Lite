import os
import json
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================
# SARVAM AI CONFIG (India-based, free tier)
# ============================================================
client = OpenAI(
    api_key=os.getenv("SARVAM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.sarvam.ai/v1"),
)

FALLBACK_MODELS = [
    "sarvam-105b",
    "sarvam-30b",
    "sarvam-m",
]

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_key(prompt, system_prompt="", temperature=0.7):
    raw = f"{system_prompt}||{prompt}||{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cache(key):
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        age_hours = (datetime.now() - datetime.fromisoformat(data["timestamp"])).total_seconds() / 3600
        if age_hours < 24:
            return data["response"]
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
    return None


def _save_cache(key, response):
    try:
        path = CACHE_DIR / f"{key}.json"
        path.write_text(json.dumps({"timestamp": datetime.now().isoformat(), "response": response}))
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


def _extract_json(text):
    """Extract JSON from LLM response, tolerating markdown fences and prose."""
    if not text:
        return text
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return match.group(0)
        except json.JSONDecodeError:
            pass
    return text


def _try_call(model, prompt, system_prompt, temperature, use_json_mode):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "timeout": 45,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_llm(prompt, system_prompt="You are a tactical military AI assistant. Respond in valid JSON only.", temperature=0.7, use_cache=True):
    cache_key = _cache_key(prompt, system_prompt, temperature)
    if use_cache:
        cached = _load_cache(cache_key)
        if cached:
            logger.info(f"Cache HIT: {cache_key}")
            return cached

    last_error = None
    for model in FALLBACK_MODELS:
        for use_json_mode in [True, False]:
            try:
                logger.info(f"Trying model={model} json_mode={use_json_mode}")
                text = _try_call(model, prompt, system_prompt, temperature, use_json_mode)
                cleaned = _extract_json(text)
                json.loads(cleaned)  # validate
                _save_cache(cache_key, cleaned)
                logger.info(f"SUCCESS: {model} (json_mode={use_json_mode})")
                return cleaned
            except json.JSONDecodeError as e:
                logger.warning(f"{model} json_mode={use_json_mode} -> non-JSON: {str(e)[:120]}")
                last_error = f"Invalid JSON: {str(e)[:200]}"
                continue
            except Exception as e:
                logger.warning(f"{model} json_mode={use_json_mode} -> error: {str(e)[:200]}")
                last_error = str(e)[:200]
                continue

    logger.error(f"All models failed. Last error: {last_error}")
    return json.dumps({"error": "LLM_UNAVAILABLE", "mock": True, "detail": last_error or "unknown"})