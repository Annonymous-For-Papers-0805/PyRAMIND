"""HotpotQA loader (distractor dev set).

Each item in the distractor dev file is:
    {
      "_id": "...",
      "question": "...",
      "answer": "...",
      "type": "comparison" | "bridge",
      "level": "easy" | "medium" | "hard",
      "supporting_facts": [[title, sent_idx], ...],
      "context": [[title, [sent1, sent2, ...]], ...]   # 10 paragraphs
    }

We treat the 10 context paragraphs as the "haystack" the system has to
ingest and reason over.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class HotpotItem:
    question_id: str
    question: str
    answer: str
    qtype: str            # "comparison" | "bridge"
    level: str            # "easy" | "medium" | "hard"
    context: list = field(default_factory=list)  # [[title, [sentences]], ...]
    supporting_facts: list = field(default_factory=list)

    def flatten_messages(self, min_len: int = 1) -> list:
        """Flatten the 10 context paragraphs into per-sentence "messages".

        Format: "<title>: <sentence>" — gives the memory engine both the
        topic anchor and the sentence content. Mirrors how mem0 / memmachine
        handle multi-paragraph QA: each sentence is one ingestible chunk.
        """
        out = []
        for entry in self.context:
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            title, sents = entry[0], entry[1]
            if not isinstance(sents, list):
                continue
            for s in sents:
                s = (s or "").strip()
                if len(s) < min_len:
                    continue
                out.append(f"{title}: {s}")
        return out


def load_hotpot(
    path: Union[Path, str],
    max_items: Optional[int] = None,
    level: Optional[str] = None,
) -> list:
    """Load HotpotQA distractor items from JSON.

    Args:
        path: dev set JSON file
        max_items: truncate after parsing
        level: filter to a specific difficulty ("easy"/"medium"/"hard")
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array, got {type(raw).__name__}")

    items: list = []
    for d in raw:
        if level is not None and str(d.get("level", "")) != level:
            continue
        items.append(HotpotItem(
            question_id=str(d.get("_id", "")),
            question=str(d.get("question", "")),
            answer=str(d.get("answer", "")),
            qtype=str(d.get("type", "")),
            level=str(d.get("level", "")),
            context=list(d.get("context", [])),
            supporting_facts=list(d.get("supporting_facts", [])),
        ))

    if max_items is not None and max_items > 0:
        items = items[:max_items]
    return items
