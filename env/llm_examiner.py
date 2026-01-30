# env/llm_examiner.py
from __future__ import annotations
import os, json, time, urllib.request, urllib.error
from typing import Any, Dict, Optional


DEFAULT_SCHEMA = {
    "name": "demand_annotation",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "urgency":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "privacy":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "complexity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["urgency", "privacy", "complexity", "confidence"]
    },
    "strict": True
}


SYSTEM_PROMPT = """You are the Examiner agent.
Given a structured summary (JSON) of ONE ICU record/window, output normalized demand scores in [0,1]:
- urgency: how time-critical / acute
- privacy: sensitivity / compliance risk
- complexity: expected compute/processing complexity
Also output confidence in [0,1].
Return ONLY the JSON object that matches the provided JSON schema.
If evidence is insufficient, be conservative and lower confidence.
"""


class GPTExaminer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout_s: int = 60,
        max_retries: int = 4,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required.")
        self.model = str(model)
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)

    def annotate(
        self,
        summary: Dict[str, Any],
        schema: Dict[str, Any] = DEFAULT_SCHEMA,
        temperature: float = 0.0,
    ) -> Dict[str, float]:
        """
        Returns: {"urgency":..,"privacy":..,"complexity":..,"confidence":..} in [0,1]
        Uses Responses API with Structured Outputs (json_schema).
        """
        url = "https://api.openai.com/v1/responses"
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": json.dumps(summary, ensure_ascii=False)}],
                },
            ],
            "temperature": float(temperature),
            "text": {
                "format": {
                    "type": "json_schema",
                    "json_schema": schema
                }
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_err = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url=url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                obj = json.loads(raw)

                # Responses API typical path: output[0].content[0].text
                txt = None
                out = obj.get("output", [])
                if out and isinstance(out, list):
                    # find first message-like item containing content text
                    for item in out:
                        content = item.get("content", [])
                        if content and isinstance(content, list):
                            for c in content:
                                if c.get("type") == "output_text" and "text" in c:
                                    txt = c["text"]
                                    break
                        if txt:
                            break

                if txt is None:
                    # fallback: some SDKs return output_text at top-level
                    txt = obj.get("output_text", None)

                if txt is None:
                    raise RuntimeError(f"Cannot find structured output text in response keys={list(obj.keys())}")

                data = json.loads(txt)
                return {
                    "urgency": float(data["urgency"]),
                    "privacy": float(data["privacy"]),
                    "complexity": float(data["complexity"]),
                    "confidence": float(data.get("confidence", 0.5)),
                }
            except Exception as e:
                last_err = e
                # exponential backoff
                time.sleep(min(2 ** attempt, 8))
                continue

        raise RuntimeError(f"GPTExaminer failed after retries: {last_err}")
