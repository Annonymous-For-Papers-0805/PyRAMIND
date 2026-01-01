"""LoCoMo dataset loader.

Format (snap-research/locomo `locomo10.json`):
- Top-level list of conversations.
- Each conversation has: speaker_a, speaker_b, conversation (with session_1,
  session_2, ... keys, each a list of utterance dicts), qa (list of QA pairs).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class LoCoMoQA:
    question: str
    answer: str
    category: int = 0
    evidence: list = field(default_factory=list)


@dataclass
class LoCoMoItem:
    conversation_id: str
    sessions: dict = field(default_factory=dict)  # {session_n: [utterances]}
    qa: list = field(default_factory=list)        # list[LoCoMoQA]

    def flatten_messages(self, min_len: int = 15) -> list:
        """Flatten all sessions in numerical order into a list of utterance strings.

        Each utterance is prefixed with "[<session-date>] <speaker>:" so date-based
        and attribution-based questions can be answered. This matches the mem0/zep
        evaluation protocol on LoCoMo (turns include date + speaker).
        """
        out = []
        keys = sorted(
            [k for k in self.sessions if re.match(r"^session_\d+$", k)],
            key=lambda k: int(k.replace("session_", "")),
        )
        for k in keys:
            date = self.sessions.get(f"{k}_date_time", "")
            prefix_date = f"[{date}] " if date else ""
            val = self.sessions[k]
            if isinstance(val, list):
                for utt in val:
                    if not isinstance(utt, dict):
                        continue
                    text = (utt.get("text") or utt.get("content") or "").strip()
                    speaker = (utt.get("speaker") or "").strip()
                    if len(text) < min_len:
                        continue
                    speaker_prefix = f"{speaker}: " if speaker else ""
                    out.append(f"{prefix_date}{speaker_prefix}{text}")
            elif isinstance(val, str):
                for line in val.split("\n"):
                    if len(line.strip()) >= min_len:
                        out.append(f"{prefix_date}{line.strip()}")
        return out


def load_locomo(
    path: Union[Path, str],
    max_conversations: Optional[int] = None,
    split: str = "all",
) -> list:
    """Load LoCoMo conversations from JSON file.

    Paper §4 fixes the LoCoMo split convention as ``dev = first 3
    conversations (≈586 questions)`` and ``test = the remaining 7
    conversations (≈1400 questions)``. ``split`` selects which slice to
    return:

    - ``"all"``  — every conversation in file order (default; back-compat).
    - ``"dev"``  — the first 3 conversations only.
    - ``"test"`` — the remaining conversations (everything after the first 3).

    ``max_conversations`` is applied AFTER the split, so smoke runs against
    e.g. ``--split dev --max-items 1`` work as expected.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raw = [raw]
    items = [_to_item(c, idx) for idx, c in enumerate(raw)]

    if split == "all":
        pass
    elif split == "dev":
        items = items[:3]
    elif split == "test":
        items = items[3:]
    else:
        raise ValueError(
            f"Unknown LoCoMo split={split!r}. Use one of 'all', 'dev', 'test'."
        )

    if max_conversations is not None and max_conversations > 0:
        items = items[:max_conversations]
    return items


def _to_item(c: dict, default_id: int) -> LoCoMoItem:
    convo = c.get("conversation", {}) or {}
    sessions = {k: v for k, v in convo.items() if k.startswith("session_")}
    qa_raw = c.get("qa", []) or []
    qa = [
        LoCoMoQA(
            question=str(q.get("question", "")),
            answer=str(q.get("answer", "")),
            category=int(q.get("category", 0)),
            evidence=list(q.get("evidence", [])),
        )
        for q in qa_raw
    ]
    return LoCoMoItem(
        conversation_id=str(c.get("sample_id", default_id)),
        sessions=sessions,
        qa=qa,
    )
