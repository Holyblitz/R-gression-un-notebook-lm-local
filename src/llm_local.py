import os, requests

def chat_ollama(model: str, system: str, user: str, json_mode: bool = False, temperature: float = 0.2, timeout: int = 120):
    """Return content string from a local Ollama chat.
    - Forces stream=False for non-streaming responses.
    - Falls back to /api/generate if /api/chat fails.
    - Returns empty string on hard failure (caller can degrade gracefully).
    """
    model = model or os.getenv("LLM_MODEL", "mistral:instruct")
    url_chat = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
    url_gen  = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate"

    body_chat = {
        "model": model,
        "messages": [
            {"role":"system","content":system or ""},
            {"role":"user","content":user or ""}
        ],
        "options": {"temperature": float(temperature)},
        "stream": False
    }
    if json_mode:
        body_chat["format"] = "json"

    try:
        r = requests.post(url_chat, json=body_chat, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message",{}).get("content","")
    except Exception:
        try:
            # Fallback to /api/generate (concatenate prompts)
            prompt = (system + "\n\n" if system else "") + (user or "")
            body_gen = {
                "model": model,
                "prompt": prompt,
                "options": {"temperature": float(temperature)},
                "stream": False
            }
            if json_mode:
                body_gen["format"] = "json"
            r2 = requests.post(url_gen, json=body_gen, timeout=timeout)
            r2.raise_for_status()
            data2 = r2.json()
            return data2.get("response","")
        except Exception:
            return ""
