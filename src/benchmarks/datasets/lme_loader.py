"""LongMemEval-S dataset loader.

The LME dataset format (per `xiaowu0162/longmemeval-cleaned`):
- Each item is a JSON object with: question_id, question_type, question, answer,
  haystack_sessions (list of session dicts with role/content turns).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class LMEItem:
    """One LongMemEval evaluation item."""

    question_id: str
    question_type: str
    question: str
    answer: str
    haystack_sessions: list = field(default_factory=list)
    haystack_dates: list = field(default_factory=list)

    def flatten_messages(self, min_len: int = 10) -> list:
        """Flatten haystack_sessions into a list of message strings.

        Each message is prefixed with "[<session-date>] <role>:" so date-based
        and attribution-based questions can be answered. Matches the LongMemEval
        eval protocol (turns include date + role).
        """
        out = []
        for i, session in enumerate(self.haystack_sessions):
            date = self.haystack_dates[i] if i < len(self.haystack_dates) else ""
            prefix_date = f"[{date}] " if date else ""
            if isinstance(session, list):
                for msg in session:
                    if not isinstance(msg, dict):
                        continue
                    content = (msg.get("content") or "").strip()
                    role = (msg.get("role") or "").strip()
                    if len(content) < min_len:
                        continue
                    role_prefix = f"{role}: " if role else ""
                    out.append(f"{prefix_date}{role_prefix}{content}")
            elif isinstance(session, dict):
                content = (session.get("content") or "").strip()
                if len(content) >= min_len:
                    out.append(f"{prefix_date}{content}")
        return out


def load_lme(
    path: Union[Path, str],
    max_items: Optional[int] = None,
) -> list:
    """Load LME items from a JSON or JSONL file.

    Returns a list of LMEItem. If `max_items` is set, truncates after parsing.
    """
    path = Path(path)
    text = path.read_text()
    items: list = []

    # Try JSON list first; fall back to JSONL.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            items = [_to_item(d) for d in data]
        else:
            items = [_to_item(data)]
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(_to_item(d))

    if max_items is not None and max_items > 0:
        items = items[:max_items]
    return items


def _to_item(d: dict) -> LMEItem:
    return LMEItem(
        question_id=str(d.get("question_id", d.get("id", ""))),
        question_type=str(d.get("question_type", "unknown")),
        question=str(d.get("question", "")),
        answer=str(d.get("answer", "")),
        haystack_sessions=list(d.get("haystack_sessions", [])),
        haystack_dates=list(d.get("haystack_dates", [])),
    )
