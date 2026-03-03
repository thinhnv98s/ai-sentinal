"""
RUN LOGGER
==========
Ghi lại toàn bộ phiên chạy vào file Markdown để review chất lượng.
"""

import json
import os
import threading
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RunMarkdownLogger:
    """Collect and export one run report as Markdown."""

    def __init__(self):
        self._lock = threading.RLock()
        self.output_dir = os.environ.get(
            "SENTINEL_RUN_LOG_DIR",
            os.path.join(os.getcwd(), "logs", "run_reports"),
        )
        self._reset_state()

    def _reset_state(self):
        self.active = False
        self.run_id = ""
        self.started_at = ""
        self.ended_at = ""
        self.watchlist: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.agent_actions: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        self.status = "UNKNOWN"
        self.error = ""

    def start_run(self, watchlist: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
        with self._lock:
            self._reset_state()
            self.active = True
            now = datetime.now()
            self.run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.started_at = now.isoformat()
            self.watchlist = list(watchlist or [])
            self.metadata = self._to_json_safe(metadata or {})
            self.log_event("RUN_START", {"watchlist": self.watchlist, "metadata": self.metadata})

    def log_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None):
        with self._lock:
            if not self.active:
                return
            self.events.append({
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "payload": self._to_json_safe(payload or {}),
            })

    def log_llm_call(self,
                     role: str,
                     model: str,
                     request_payload: Dict[str, Any],
                     response_payload: Any,
                     symbol: str = "",
                     status: str = "SUCCESS",
                     latency_ms: Optional[float] = None,
                     error: str = "",
                     attempt: int = 1):
        with self._lock:
            if not self.active:
                return
            self.llm_calls.append({
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "model": model,
                "symbol": symbol,
                "status": status,
                "latency_ms": latency_ms,
                "error": error,
                "attempt": attempt,
                "request": self._to_json_safe(request_payload),
                "response": self._to_json_safe(response_payload),
            })

    def log_blackboard_message(self, message_dict: Dict[str, Any]):
        with self._lock:
            if not self.active:
                return
            self.agent_actions.append(self._to_json_safe(message_dict))

    def finalize(self,
                 status: str = "COMPLETED",
                 summary: Optional[Dict[str, Any]] = None,
                 error: str = "") -> Optional[str]:
        with self._lock:
            if not self.active:
                return None

            self.ended_at = datetime.now().isoformat()
            self.status = status
            self.error = error
            self.summary = self._to_json_safe(summary or {})
            self.log_event("RUN_END", {"status": status, "error": error, "summary": self.summary})

            os.makedirs(self.output_dir, exist_ok=True)
            file_path = os.path.join(self.output_dir, f"{self.run_id}.md")

            content = self._build_markdown()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.active = False
            return file_path

    def _build_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# Sentinel Run Report - {self.run_id}")
        lines.append("")
        lines.append("## Run Metadata")
        lines.append(f"- Status: {self.status}")
        lines.append(f"- Started At: {self.started_at}")
        lines.append(f"- Ended At: {self.ended_at}")
        lines.append(f"- Watchlist: {', '.join(self.watchlist) if self.watchlist else 'N/A'}")
        if self.error:
            lines.append(f"- Error: {self.error}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(self.metadata, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        lines.append("## Run Summary")
        lines.append("```json")
        lines.append(json.dumps(self.summary, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        lines.append("## Timeline Events")
        if not self.events:
            lines.append("_No events captured._")
        for idx, evt in enumerate(self.events, start=1):
            lines.append(f"### Event {idx}: {evt.get('event_type', 'UNKNOWN')}")
            lines.append(f"- Timestamp: {evt.get('timestamp', '')}")
            lines.append("```json")
            lines.append(json.dumps(evt.get("payload", {}), indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")

        lines.append("## LLM Request-Response")
        if not self.llm_calls:
            lines.append("_No LLM calls captured._")
        for idx, call in enumerate(self.llm_calls, start=1):
            lines.append(
                f"### LLM Call {idx}: role={call.get('role', '')} model={call.get('model', '')} "
                f"status={call.get('status', '')} attempt={call.get('attempt', 1)}"
            )
            lines.append(f"- Timestamp: {call.get('timestamp', '')}")
            lines.append(f"- Symbol: {call.get('symbol', '') or 'N/A'}")
            lines.append(f"- Latency(ms): {call.get('latency_ms', '')}")
            if call.get("error"):
                lines.append(f"- Error: {call.get('error')}")
            lines.append("#### Request")
            lines.append("```json")
            lines.append(json.dumps(call.get("request", {}), indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("#### Response")
            lines.append("```json")
            lines.append(json.dumps(call.get("response", {}), indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")

        lines.append("## Agent Actions (Blackboard Messages)")
        if not self.agent_actions:
            lines.append("_No agent actions captured._")
        for idx, msg in enumerate(self.agent_actions, start=1):
            sender = msg.get("sender", "UNKNOWN")
            receiver = msg.get("receiver", "UNKNOWN")
            msg_type = msg.get("msg_type", "UNKNOWN")
            lines.append(f"### Action {idx}: {sender} -> {receiver} [{msg_type}]")
            lines.append(f"- Timestamp: {msg.get('timestamp', '')}")
            lines.append(f"- Priority: {msg.get('priority', 'NORMAL')}")
            lines.append("```json")
            lines.append(json.dumps(msg.get("content", {}), indent=2, ensure_ascii=False))
            lines.append("```")
        lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _to_json_safe(self, value: Any, depth: int = 0) -> Any:
        if depth > 12:
            return str(value)

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Enum):
            return value.value

        if is_dataclass(value):
            return self._to_json_safe(asdict(value), depth + 1)

        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v, depth + 1) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._to_json_safe(v, depth + 1) for v in value]

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self._to_json_safe(value.to_dict(), depth + 1)
            except Exception:
                return str(value)

        if hasattr(value, "__dict__"):
            try:
                return self._to_json_safe(vars(value), depth + 1)
            except Exception:
                return str(value)

        return str(value)


_run_logger_instance: Optional[RunMarkdownLogger] = None
_run_logger_lock = threading.Lock()


def get_run_logger() -> RunMarkdownLogger:
    global _run_logger_instance
    with _run_logger_lock:
        if _run_logger_instance is None:
            _run_logger_instance = RunMarkdownLogger()
        return _run_logger_instance
